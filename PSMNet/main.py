from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import time
import math
from dataloader import CustomLoader as CL
from models import *

parser = argparse.ArgumentParser(description='PSMNet – custom dataset training')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model: stackhourglass | basic')
parser.add_argument('--datapath', required=True,
                    help='path to dataset root folder '
                         '(must contain left/, right/, disparity/ sub-folders)')
parser.add_argument('--crop-h', type=int, default=384,
                    help='random crop height for training (must be multiple of 16, default: 384)')
parser.add_argument('--crop-w', type=int, default=768,
                    help='random crop width for training (must be multiple of 16, default: 768)')
parser.add_argument('--val-split', type=float, default=0.2,
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=4,
                    help='training batch size (default: 4; adjust to GPU memory)')
parser.add_argument('--loadmodel', default=None,
                    help='path to pretrained model checkpoint (.tar)')
parser.add_argument('--savemodel', default='./',
                    help='directory to save checkpoints')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# ── GPU performance flags ─────────────────────────────────────────────────────
if args.cuda:
    cudnn.benchmark = True                         # auto-select fastest conv algorithm
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmul (~3× speed on Blackwell)
    cudnn.allow_tf32 = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ── Dataset & DataLoaders ─────────────────────────────────────────────────────
# Expects datapath/left/, datapath/right/, datapath/disparity/
# Files named 000000.png, 000001.png, ...
# Images are fed at full resolution (384 × 768) — no cropping applied.
train_left, train_right, train_disp, \
    val_left, val_right, val_disp = CL.dataloader(args.datapath,
                                                   val_split=args.val_split)

TrainImgLoader = torch.utils.data.DataLoader(
    CL.CustomDataset(train_left, train_right, train_disp, training=True,
                     crop_h=args.crop_h, crop_w=args.crop_w),
    batch_size=args.batch_size, shuffle=True,
    num_workers=8, drop_last=True, pin_memory=True)

ValImgLoader = torch.utils.data.DataLoader(
    CL.CustomDataset(val_left, val_right, val_disp, training=False),
    batch_size=max(1, args.batch_size // 2), shuffle=False,
    num_workers=4, drop_last=False, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    raise ValueError(f"Unknown model: {args.model}. Choose 'stackhourglass' or 'basic'.")

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model from:', args.loadmodel)
    pretrain_dict = torch.load(args.loadmodel, weights_only=False)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {:,}'.format(
    sum(p.data.nelement() for p in model.parameters())))

os.makedirs(args.savemodel, exist_ok=True)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Mixed-precision scaler — BF16 autocast on Blackwell gives significant speedup
scaler = torch.amp.GradScaler('cuda')


# ── Training step ─────────────────────────────────────────────────────────────
def train(imgL, imgR, disp_L):
    model.train()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    else:
        disp_true = disp_L

    # Only train on pixels with valid (> 0) and in-range disparity
    mask = (disp_true > 0) & (disp_true < args.maxdisp)
    mask.detach_()

    optimizer.zero_grad()

    with torch.amp.autocast('cuda'):
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL, imgR)
            output1 = torch.squeeze(output1, 1)
            output2 = torch.squeeze(output2, 1)
            output3 = torch.squeeze(output3, 1)
            loss = (0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean')
                  + 0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], reduction='mean')
                  +       F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean'))
        elif args.model == 'basic':
            output = model(imgL, imgR)
            output = torch.squeeze(output, 1)
            loss = F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


# ── Validation step ───────────────────────────────────────────────────────────
def validate(imgL, imgR, disp_true, pad_h=0, pad_w=0):
    model.eval()

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        output = model(imgL, imgR)
        output = torch.squeeze(output)   # (H_padded, W_padded)

    # Strip padding that was added by the dataset to make dims divisible by 16
    if pad_h > 0:
        output     = output[..., :output.shape[-2] - pad_h, :]
        disp_true  = disp_true[..., :disp_true.shape[-2] - pad_h, :]
    if pad_w > 0:
        output     = output[..., :output.shape[-1] - pad_w]
        disp_true  = disp_true[..., :disp_true.shape[-1] - pad_w]

    mask = (disp_true > 0) & (disp_true < args.maxdisp)
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return F.l1_loss(output[mask], disp_true[mask]).cpu()


# ── Learning rate schedule ────────────────────────────────────────────────────
def adjust_learning_rate(optimizer, epoch):
    # Decay LR by 10× at epoch 6 (original PSMNet schedule adapted for short runs)
    lr = 0.001 if epoch < 6 else 0.0001
    print(f'  LR: {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    start_full_time = time.time()

    for epoch in range(0, args.epochs):
        print(f'\n======== Epoch {epoch} ========')
        adjust_learning_rate(optimizer, epoch)
        total_train_loss = 0.0

        # ── Train ──────────────────────────────────────────────────────────
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TrainImgLoader):
            t0   = time.time()
            loss = train(imgL, imgR, disp_L)
            total_train_loss += loss
            if batch_idx % 10 == 0:
                print(f'  [Train] iter {batch_idx:4d}  loss={loss:.4f}  '
                      f'time={time.time()-t0:.2f}s')

        avg_train_loss = total_train_loss / len(TrainImgLoader)
        print(f'  [Train] epoch {epoch} avg loss = {avg_train_loss:.4f}')

        # ── Validate ───────────────────────────────────────────────────────
        total_val_loss = 0.0
        for batch_idx, (imgL, imgR, disp_L, pad_h, pad_w) in enumerate(ValImgLoader):
            val_loss = validate(imgL, imgR, disp_L,
                                pad_h=int(pad_h[0]), pad_w=int(pad_w[0]))
            total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / max(len(ValImgLoader), 1)
        print(f'  [Val]   epoch {epoch} avg EPE  = {avg_val_loss:.4f} px')

        # ── Save checkpoint ────────────────────────────────────────────────
        ckpt_path = os.path.join(args.savemodel, f'checkpoint_{epoch:04d}.tar')
        torch.save({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
        }, ckpt_path)
        print(f'  Checkpoint saved → {ckpt_path}')

    elapsed = (time.time() - start_full_time) / 3600
    print(f'\nFull training time: {elapsed:.2f} hr')


if __name__ == '__main__':
    main()

# python main.py \
#     --datapath C:/path/to/your/dataset \
#     --model stackhourglass \
#     --maxdisp 128 \
#     --epochs 10 \
#     --batch-size 4 \
#     --savemodel ./checkpoints
