"""
CustomLoader.py
---------------
Dataloader for a custom stereo dataset with the following structure:

    datapath/
        left/       000000.png, 000001.png, ...
        right/      000000.png, 000001.png, ...
        disparity/  000000.png, 000001.png, ...   (or .pfm)

Disparity convention
--------------------
  - 16-bit PNG  → pixel_value / 256.0  (matches KITTI convention)
  - PFM         → read directly as float
  - 8-bit PNG   → pixel_value as-is  (raw pixel = disparity in pixels)

Images are loaded at their native resolution and **not** randomly cropped,
so they are fed to the network at full size (e.g. 384 × 768).
The resolution must be divisible by 16 on both dimensions.
"""

import os
import re
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from . import readpfm as rp

# ── Image statistics (ImageNet) ───────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.ppm', '.pfm', ''}  # '' = no extension


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sorted_files(folder: str) -> list:
    """Return sorted list of absolute paths for all files in folder.

    Accepts files with known image extensions (.png, .jpg, .pfm …)
    AND extension-less files (e.g. named 000000, 000001, …).
    PIL will detect the actual format from the file header at load time.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    if not files:
        raise RuntimeError(f"No image files found in: {folder}")
    return files


def _load_disparity(path: str) -> np.ndarray:
    """Load a disparity map and return a float32 numpy array (H × W).

    Format detection is done via PIL's header-sniffing, not the filename
    extension, so extension-less files (000000, 000001 …) are handled fine.

    16-bit PNG (mode 'I' or 'I;16', max > 255) → divide by 256.0 (KITTI).
    8-bit PNG (mode 'L', values 0-255)          → raw value = disparity px.
    PFM (extension .pfm)                         → read float directly.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pfm':
        disp, _ = rp.readPFM(path)
        return np.ascontiguousarray(disp, dtype=np.float32)

    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    # 16-bit grayscale: PIL reports mode 'I' (or raw 'I;16')
    if img.mode in ('I', 'I;16') or arr.max() > 255.0:
        arr = arr / 256.0   # KITTI convention: stored as disparity × 256
    # 8-bit (mode 'L'): raw pixel value equals disparity in pixels
    return np.ascontiguousarray(arr, dtype=np.float32)


# ── Dataset listing ──────────────────────────────────────────────────────────

def dataloader(datapath: str, val_split: float = 0.2):
    """
    Scan datapath/{left,right,disparity} and return
    (train_left, train_right, train_disp, val_left, val_right, val_disp).

    Args:
        datapath:   root folder containing left/, right/, disparity/
        val_split:  fraction of data to use for validation (default 0.2)

    Returns:
        Six lists of file paths.
    """
    left_dir  = os.path.join(datapath, 'left')
    right_dir = os.path.join(datapath, 'right')
    disp_dir  = os.path.join(datapath, 'disparity')

    left_files  = _sorted_files(left_dir)
    right_files = _sorted_files(right_dir)
    disp_files  = _sorted_files(disp_dir)

    n = len(left_files)
    assert len(right_files) == n, \
        f"Mismatch: {len(left_files)} left vs {len(right_files)} right images"
    assert len(disp_files)  == n, \
        f"Mismatch: {len(left_files)} left vs {len(disp_files)} disparity maps"

    split_idx = int(n * (1.0 - val_split))

    train_left  = left_files[:split_idx]
    train_right = right_files[:split_idx]
    train_disp  = disp_files[:split_idx]

    val_left    = left_files[split_idx:]
    val_right   = right_files[split_idx:]
    val_disp    = disp_files[split_idx:]

    print(f"[CustomLoader] Found {n} samples → "
          f"{len(train_left)} train / {len(val_left)} val")

    return train_left, train_right, train_disp, val_left, val_right, val_disp


# ── Dataset class ────────────────────────────────────────────────────────────

def _pad16(img_tensor: torch.Tensor, disp_tensor: torch.Tensor):
    """Pad H and W up to the nearest multiple of 16 (bottom / right padding).
    Returns (padded_img, padded_disp, pad_h, pad_w) so the caller can crop
    the prediction back to the original size."""
    _, h, w = img_tensor.shape
    pad_h = (16 - h % 16) % 16   # 0 if already divisible
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        import torch.nn.functional as F
        img_tensor  = F.pad(img_tensor,  (0, pad_w, 0, pad_h))
        disp_tensor = F.pad(disp_tensor, (0, pad_w, 0, pad_h))
    return img_tensor, disp_tensor, pad_h, pad_w


class CustomDataset(data.Dataset):
    """
    Stereo dataset for custom root/left + root/right + root/disparity layout.

    Training  : random crop of size crop_h × crop_w (both must be multiples of 16).
    Validation: full image padded to the nearest multiple of 16; the caller is
                expected to strip the padding from the network output.

    Default crop: 384 × 768  (divisible by 16, reasonable for 900 × 1600 images).
    """

    def __init__(self, left_files, right_files, disp_files, training: bool,
                 crop_h: int = 384, crop_w: int = 768):
        super().__init__()
        assert crop_h % 16 == 0 and crop_w % 16 == 0, \
            f"crop_h ({crop_h}) and crop_w ({crop_w}) must both be multiples of 16."
        self.left_files  = left_files
        self.right_files = right_files
        self.disp_files  = disp_files
        self.training    = training
        self.crop_h      = crop_h
        self.crop_w      = crop_w

    def __len__(self):
        return len(self.left_files)

    def __getitem__(self, idx):
        import random as _random
        left_img  = Image.open(self.left_files[idx]).convert('RGB')
        right_img = Image.open(self.right_files[idx]).convert('RGB')
        disp      = _load_disparity(self.disp_files[idx])   # (H, W) float32

        w, h = left_img.size   # PIL gives (width, height)

        if self.training:
            # ── Random crop ───────────────────────────────────────────────────
            assert h >= self.crop_h and w >= self.crop_w, (
                f"Image {h}×{w} is smaller than the requested crop "
                f"{self.crop_h}×{self.crop_w}.")
            x0 = _random.randint(0, w - self.crop_w)
            y0 = _random.randint(0, h - self.crop_h)
            left_img  = left_img.crop((x0, y0, x0 + self.crop_w, y0 + self.crop_h))
            right_img = right_img.crop((x0, y0, x0 + self.crop_w, y0 + self.crop_h))
            disp      = disp[y0:y0 + self.crop_h, x0:x0 + self.crop_w]

            left_t  = _img_transform(left_img)     # (3, crop_h, crop_w)
            right_t = _img_transform(right_img)
            disp_t  = torch.from_numpy(disp)       # (crop_h, crop_w)
            return left_t, right_t, disp_t

        else:
            # ── Validation: pad to nearest multiple of 16 ─────────────────────
            left_t  = _img_transform(left_img)     # (3, H, W)
            right_t = _img_transform(right_img)
            disp_t  = torch.from_numpy(disp)       # (H, W)

            left_t, disp_t, pad_h, pad_w = _pad16(left_t, disp_t)
            right_t, _,     _,     _     = _pad16(right_t, disp_t)

            # Store padding amounts in the disparity tensor's metadata via a
            # wrapper tuple so main.py can strip them from the prediction.
            return left_t, right_t, disp_t, pad_h, pad_w
