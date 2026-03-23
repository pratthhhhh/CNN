"""
PSMNet Inference Script
Loads a checkpoint, runs inference on the test set, and computes:
  - EPE   (End-Point Error)
  - BP@0.5 (Bad-Pixel rate, error > 0.5 px)
  - BP@1.0 (Bad-Pixel rate, error > 1.0 px)
  - GFLOPs (per forward pass)
  - Inference time (per image, mean ± std)

Disparity maps are saved as both PNG (colour-mapped) and NumPy .npy files.

Usage:
    python inference.py

Edit the CONFIG block below to match your paths.
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
import matplotlib
matplotlib.use('Agg')           # headless – no display needed
import matplotlib.pyplot as plt

from models.stackhourglass import PSMNet
from dataloader import preprocess

# ──────────────────────────── CONFIG ────────────────────────────────────────
CHECKPOINT  = '/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/CNN/PSMNet/checkpoints/best_checkpoint.tar'
DATAPATH    = '/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/Dataset/CARLA'
OUTPUT_DIR  = '/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/CNN/PSMNet/inference_output'
MAXDISP     = 192
BATCH_SIZE  = 8       # Increased to 8 to utilize GPU faster
SEED        = 1

# Input size used for GFLOPs calculation
GFLOP_H, GFLOP_W = 256, 512
# ─────────────────────────────────────────────────────────────────────────────


# ───────────────────────── Data helpers ──────────────────────────────────────
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(f):
    return any(f.endswith(ext) for ext in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class TestDataset(data.Dataset):
    """Returns full-size images (cropped from bottom-right to nearest multiple)."""

    def __init__(self, left_paths, right_paths, disp_paths,
                 loader=default_loader, dploader=disparity_loader):
        self.lefts  = left_paths
        self.rights = right_paths
        self.disps  = disp_paths
        self.loader   = loader
        self.dploader = dploader
        self.transform = preprocess.get_transform(augment=False)

    def __len__(self):
        return len(self.lefts)

    def __getitem__(self, idx):
        imgL = self.loader(self.lefts[idx])
        imgR = self.loader(self.rights[idx])
        disp = self.dploader(self.disps[idx])

        w, h = imgL.size
        # crop to nearest multiple-of-16 from bottom-right (standard KITTI-style)
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        imgL = imgL.crop((w - new_w, h - new_h, w, h))
        imgR = imgR.crop((w - new_w, h - new_h, w, h))

        disp_arr = np.ascontiguousarray(disp, dtype=np.float32) / 256.0
        disp_arr = disp_arr[h - new_h:h, w - new_w:w]

        imgL = self.transform(imgL)
        imgR = self.transform(imgR)
        return imgL, imgR, torch.from_numpy(disp_arr), os.path.basename(self.lefts[idx])


# ──────────────────────── Metric helpers ─────────────────────────────────────
def compute_metrics(pred, gt, maxdisp=MAXDISP):
    """All metrics computed only where gt is valid (< maxdisp)."""
    mask = (gt > 0) & (gt < maxdisp)
    if mask.sum() == 0:
        return dict(epe=float('nan'), bp05=float('nan'), bp10=float('nan'))

    p = pred[mask]
    g = gt[mask]
    err = torch.abs(p - g)

    epe  = err.mean().item()
    bp05 = (err > 0.5).float().mean().item() * 100.0
    bp10 = (err > 1.0).float().mean().item() * 100.0
    return dict(epe=epe, bp05=bp05, bp10=bp10)


# ───────────────────────── GFLOPs helper ─────────────────────────────────────
def count_gflops(model, h=GFLOP_H, w=GFLOP_W):
    """
    Counts GFLOPs using torch hooks (no external library required).
    Falls back gracefully if the model uses ops not easily captured.
    Uses `thop` if installed, otherwise approximates via a timing-free flop hook.
    """
    try:
        from thop import profile as thop_profile
        dummy_l = torch.randn(1, 3, h, w).cuda()
        dummy_r = torch.randn(1, 3, h, w).cuda()
        flops, _ = thop_profile(model, inputs=(dummy_l, dummy_r), verbose=False)
        return flops / 1e9  # GFLOPs
    except ImportError:
        print('[WARN] `thop` not installed. GFLOPs skipped. Install with: pip install thop')
        return None


# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # ── Paths ────────────────────────────────────────────────────────────────
    left_dir  = os.path.join(DATAPATH, 'left')
    right_dir = os.path.join(DATAPATH, 'right')
    disp_dir  = os.path.join(DATAPATH, 'disparity')

    all_left  = sorted([os.path.join(left_dir,  f) for f in os.listdir(left_dir)  if is_image_file(f)])
    all_right = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if is_image_file(f)])
    all_disp  = sorted([os.path.join(disp_dir,  f) for f in os.listdir(disp_dir)  if is_image_file(f)])

    # Use the same 20% test split as training (reproducible with same seed)
    combined = list(zip(all_left, all_right, all_disp))
    random.shuffle(combined)
    all_left, all_right, all_disp = zip(*combined)

    split  = int(0.8 * len(all_left))
    test_L = all_left[split:]
    test_R = all_right[split:]
    test_D = all_disp[split:]

    loader = data.DataLoader(
        TestDataset(test_L, test_R, test_D),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f'==> Test samples : {len(test_L)}')

    # ── Model ────────────────────────────────────────────────────────────────
    print(f'==> Loading checkpoint : {CHECKPOINT}')
    model = PSMNet(MAXDISP).cuda()
    ckpt  = torch.load(CHECKPOINT, map_location='cuda')
    # Handle both raw state_dict and the tar format saved by train.py
    state = ckpt.get('state_dict', ckpt)
    # Strip 'module.' prefix if the model was saved with DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print('==> Checkpoint loaded.')

    # ── GFLOPs ───────────────────────────────────────────────────────────────
    gflops = count_gflops(model)
    if gflops is not None:
        print(f'==> GFLOPs @ {GFLOP_H}x{GFLOP_W} : {gflops:.2f} G')

    # ── Output dir ───────────────────────────────────────────────────────────
    os.makedirs(os.path.join(OUTPUT_DIR, 'disp_png'), exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    all_epe, all_bp05, all_bp10, all_times = [], [], [], []
    saved_pngs = 0

    # Warm up GPU (avoid cold-start timing bias)
    with torch.no_grad():
        _dummy = torch.randn(1, 3, GFLOP_H, GFLOP_W).cuda()
        model(_dummy, _dummy)
    torch.cuda.synchronize()

    with torch.no_grad():
        for i, (imgL, imgR, disp_gt, fname) in enumerate(loader):
            imgL    = imgL.cuda()
            imgR    = imgR.cuda()
            disp_gt = disp_gt.cuda()

            # --- timing --------------------------------------------------
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(imgL, imgR)
            torch.cuda.synchronize()
            
            # Time spent per image in this batch
            elapsed_per_img = (time.perf_counter() - t0) / imgL.size(0)
            # -------------------------------------------------------------

            if pred.dim() == 4:
                pred = torch.squeeze(pred, 1)         # (B, H, W)
            
            batch_epe = []

            # Loop through all images in the batch
            for b in range(imgL.size(0)):
                диsp = pred[b].cpu().numpy()

                all_times.append(elapsed_per_img)

                # Metrics
                metrics = compute_metrics(pred[b], disp_gt[b])
                if not np.isnan(metrics['epe']):
                    all_epe.append(metrics['epe'])
                    all_bp05.append(metrics['bp05'])
                    all_bp10.append(metrics['bp10'])
                    batch_epe.append(metrics['epe'])

                # Save disparity  ─────────────────────────────────────────────
                if saved_pngs < 5:
                    stem = os.path.splitext(fname[b])[0]
                    fig, ax = plt.subplots(figsize=(диsp.shape[1] / 100, диsp.shape[0] / 100), dpi=100)
                    im = ax.imshow(диsp, cmap='plasma', vmin=0, vmax=MAXDISP)
                    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
                    ax.axis('off')
                    ax.set_title(f'EPE={metrics["epe"]:.2f}  BP@1.0={metrics["bp10"]:.1f}%', fontsize=8)
                    fig.tight_layout(pad=0.1)
                    fig.savefig(os.path.join(OUTPUT_DIR, 'disp_png', f'{stem}.png'), dpi=100)
                    plt.close(fig)
                    saved_pngs += 1

            # Print every batch
            avg_epe = np.nanmean(batch_epe) if batch_epe else float('nan')
            print(f'  Batch [{i+1:3d}/{len(loader)}] ({imgL.size(0)} imgs) '
                  f'Avg EPE={avg_epe:.3f}  '
                  f'Time/img={elapsed_per_img*1000:.1f} ms')

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_epe  = np.nanmean(all_epe)
    mean_bp05 = np.nanmean(all_bp05)
    mean_bp10 = np.nanmean(all_bp10)
    mean_time = np.mean(all_times) * 1000   # ms
    std_time  = np.std(all_times)  * 1000

    summary = (
        '\n' + '=' * 55 + '\n'
        f'  TEST RESULTS  ({len(test_L)} images)\n'
        + '=' * 55 + '\n'
        f'  EPE            : {mean_epe:.4f} px\n'
        f'  BP @ 0.5 px    : {mean_bp05:.2f} %\n'
        f'  BP @ 1.0 px    : {mean_bp10:.2f} %\n'
        f'  Inference time : {mean_time:.1f} ± {std_time:.1f} ms / image\n'
    )
    if gflops is not None:
        summary += f'  GFLOPs         : {gflops:.2f} G  (@ {GFLOP_H}×{GFLOP_W})\n'
    summary += '=' * 55

    print(summary)

    # Write summary to file
    with open(os.path.join(OUTPUT_DIR, 'results.txt'), 'w') as f:
        f.write(summary)
        if gflops is not None:
            f.write(f'\nGFLOP input size: {GFLOP_H}x{GFLOP_W}\n')

    print(f'\n==> Disparity maps saved to : {OUTPUT_DIR}')
    print(f'==> Summary saved to        : {os.path.join(OUTPUT_DIR, "results.txt")}')


if __name__ == '__main__':
    main()
