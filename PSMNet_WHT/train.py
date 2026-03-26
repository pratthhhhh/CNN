"""
PSMNet Training Script
Extracted from CNN.ipynb - training logic only, no visualizations.

Usage:
    python train.py

Dataset path and hyperparameters can be modified in the CONFIG section below.
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image

from models import *
from dataloader import KITTILoader as DA
from dataloader import preprocess

# ─────────────────────────── CONFIG ────────────────────────────────────────
DATAPATH    = '/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/Dataset/CARLA'
SAVEMODEL   = '/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/CNN/PSMNet/checkpoints'
MAXDISP     = 192
EPOCHS      = 50
PATIENCE    = 5
BATCH_TRAIN = 4
BATCH_TEST  = 2
LR          = 0.001
SEED        = 1
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────── Image / disparity helpers ──────────────────────────
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


# ──────────────────────────── Dataset ───────────────────────────────────────
class MyImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training,
                 loader=default_loader, dploader=disparity_loader):
        self.left    = left
        self.right   = right
        self.disp_L  = left_disparity
        self.loader  = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left    = self.left[index]
        right   = self.right[index]
        disp_L  = self.disp_L[index]

        left_img  = self.loader(left)
        right_img = self.loader(right)
        dataL     = self.dploader(disp_L)

        if self.training:
            w, h  = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img  = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img  = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL

        else:
            w, h = left_img.size
            left_img  = left_img.crop( (w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))
            dataL     = dataL.crop(   (w - 1232, h - 368, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)
            left_img  = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)


# ─────────────────────── Model setup ────────────────────────────────────────
def build_model(maxdisp=MAXDISP):
    from models.stackhourglass import PSMNet as StackPSM
    model = StackPSM(maxdisp)
    model.cuda()
    return model


# ─────────────────────── Train / Test steps ──────────────────────────────────
def train(model, optimizer, imgL, imgR, disp_L):
    model.train()

    imgL   = imgL.float().cuda()
    imgR   = imgR.float().cuda()
    disp_L = disp_L.float().cuda()

    mask = disp_L < MAXDISP
    mask.detach_()

    optimizer.zero_grad()

    output1, output2, output3 = model(imgL, imgR)
    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)

    loss = (0.5 * F.smooth_l1_loss(output1[mask], disp_L[mask], reduction='mean')
          + 0.7 * F.smooth_l1_loss(output2[mask], disp_L[mask], reduction='mean')
          +       F.smooth_l1_loss(output3[mask], disp_L[mask], reduction='mean'))

    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, imgL, imgR, disp_true):
    model.eval()

    imgL      = imgL.float().cuda()
    imgR      = imgR.float().cuda()
    disp_true = disp_true.float().cuda()

    mask = disp_true < MAXDISP
    mask.detach_()

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 1)

    loss = F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean')
    return loss.item()


# ─────────────────────── Main ────────────────────────────────────────────────
def main():
    print(f'==> Using device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})')
    print(f'==> GPUs available: {torch.cuda.device_count()}')

    # Reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────
    left_image_path   = os.path.join(DATAPATH, 'left')
    right_image_path  = os.path.join(DATAPATH, 'right')
    disparity_path    = os.path.join(DATAPATH, 'disparity')

    left_images       = sorted([os.path.join(left_image_path,  f)
                                 for f in os.listdir(left_image_path)  if is_image_file(f)])
    right_images      = sorted([os.path.join(right_image_path, f)
                                 for f in os.listdir(right_image_path) if is_image_file(f)])
    disparity_images  = sorted([os.path.join(disparity_path,   f)
                                 for f in os.listdir(disparity_path)   if is_image_file(f)])

    # Train / val split (80 / 20)
    train_size = int(0.8 * len(left_images))

    combined = list(zip(left_images, right_images, disparity_images))
    random.shuffle(combined)
    left_images, right_images, disparity_images = zip(*combined)

    train_left_img  = left_images[:train_size]
    train_right_img = right_images[:train_size]
    train_left_disp = disparity_images[:train_size]

    test_left_img   = left_images[train_size:]
    test_right_img  = right_images[train_size:]
    test_left_disp  = disparity_images[train_size:]

    TrainImgLoader = torch.utils.data.DataLoader(
        MyImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=BATCH_TRAIN, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader  = torch.utils.data.DataLoader(
        MyImageFloder(test_left_img,  test_right_img,  test_left_disp,  False),
        batch_size=BATCH_TEST,  shuffle=False, num_workers=2, drop_last=False)

    # ── Model & optimizer ─────────────────────────────────────────────────
    print(f'==> Building PSMNet (maxdisp={MAXDISP}) ...')
    model     = build_model(MAXDISP)

    # Enable TF32 on Ampere+ GPUs (~3x faster matmul/conv at ~FP16 precision)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # Compile the full model — fuses WHT butterfly + all ops into minimal
    # CUDA kernels.  First few iterations are slow (compilation warmup).
    if hasattr(torch, 'compile'):
        print('==> Compiling model with torch.compile ...')
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    print(f'==> Model ready. Training samples: {len(train_left_img)}, Val samples: {len(test_left_img)}')
    print(f'==> Starting {EPOCHS} epochs ...')

    os.makedirs(SAVEMODEL, exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch   = 1

    # ── Training loop ─────────────────────────────────────────────────────────
    start_full_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        total_train_loss = 0
        total_test_loss  = 0

        # Training
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(model, optimizer, imgL_crop, imgR_crop, disp_crop_L)
            print(f'Epoch {epoch}, Iter {batch_idx} training loss = {loss:.3f}, '
                  f'time = {time.time() - start_time:.2f}')
            total_train_loss += loss

        avg_train_loss = total_train_loss / len(TrainImgLoader)
        print(f'epoch {epoch} total training loss = {avg_train_loss:.3f}')

        # Validation
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(model, imgL, imgR, disp_L)
            print(f'Iter {batch_idx} validation loss = {test_loss:.3f}')
            total_test_loss += test_loss

        avg_test_loss = total_test_loss / len(TestImgLoader)
        print(f'epoch {epoch} total validation loss = {avg_test_loss:.3f}')

        # Save best checkpoint
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            epochs_no_improve = 0
            savefilename  = os.path.join(SAVEMODEL, 'best_checkpoint.tar')
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss':  avg_test_loss,
            }, savefilename)
            print(f'Saved best model at epoch {epoch} with val loss {avg_test_loss:.3f}')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Patience: {epochs_no_improve}/{PATIENCE}')
            if epochs_no_improve >= PATIENCE:
                print('Early stopping triggered!')
                break

    print(f'Full training time = {(time.time() - start_full_time) / 3600:.2f} HR')


if __name__ == '__main__':
    main()
