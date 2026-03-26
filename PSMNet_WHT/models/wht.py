"""
Optimized Walsh-Hadamard Transform convolution layer.

Key optimization: replaces the butterfly-loop FWHT (O(n log n) FLOPs but
~128 tiny CUDA kernel launches per layer) with cached Hadamard matrix
multiplies (O(n^2) FLOPs but only 4 fused cuBLAS calls per layer).

For the spatial sizes in PSMNet (n = 64-256 during training, up to 1024
at test time) the kernel-launch overhead of the butterfly vastly exceeds
the extra arithmetic of the matmul, so this is a net win in wall-clock time.

2D WHT  :   H_h  @  x  @  H_w
2D iWHT : ( H_h  @  x  @  H_w ) / (h_pad * w_pad)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard


# ── helpers ─────────────────────────────────────────────────────────────────

def find_min_power(x, p=2):
    """Smallest power of *p* that is >= x."""
    y = 1
    while y < x:
        y *= p
    return y


# Module-level cache so every WHTConv2D layer sharing the same padded size
# (and device) reuses the same tensor rather than allocating its own.
_H_CACHE = {}


def _cached_hadamard(n, device):
    key = (n, str(device))
    if key not in _H_CACHE:
        _H_CACHE[key] = torch.tensor(
            hadamard(n), dtype=torch.float32, device=device)
    return _H_CACHE[key]


# ── soft thresholding ───────────────────────────────────────────────────────

class SoftThresholding(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = nn.Parameter(torch.rand(self.num_features) / 10)

    def forward(self, x):
        return torch.copysign(
            F.relu(torch.abs(x) - F.relu(self.T)), x)


# ── WHTConv2D ───────────────────────────────────────────────────────────────

class WHTConv2D(nn.Module):
    """Walsh-Hadamard-Transform 2-D convolution (matmul backend).

    Construction args are identical to the original butterfly version so that
    submodule.py needs no changes.
    """

    def __init__(self, height, width, in_channels, out_channels,
                 pods=1, residual=True):
        super().__init__()
        self.height     = height
        self.width      = width
        self.height_pad = find_min_power(height)
        self.width_pad  = find_min_power(width)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.pods     = pods
        self.residual = residual

        # 1x1 channel mixing
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            for _ in range(pods)])

        # Spectral-domain learnable parameters
        self.ST = nn.ModuleList([
            SoftThresholding((self.height_pad, self.width_pad))
            for _ in range(pods)])
        self.v = nn.ParameterList([
            nn.Parameter(torch.rand(self.height_pad, self.width_pad))
            for _ in range(pods)])

        # ── cached Hadamard matrices (non-learnable, move with .cuda()) ────
        self.register_buffer(
            'H_h', torch.tensor(hadamard(self.height_pad), dtype=torch.float32))
        self.register_buffer(
            'H_w', torch.tensor(hadamard(self.width_pad),  dtype=torch.float32))

    # ------------------------------------------------------------------

    def _get_hadamard_pair(self, h_pad, w_pad):
        """Return (H_h, H_w) for the given padded sizes.

        Training path (sizes match __init__): returns registered buffers
        (already on the correct device, zero overhead).

        Inference path (different image size): fetches from a global cache
        (created once per unique (size, device) pair).
        """
        if h_pad == self.height_pad and w_pad == self.width_pad:
            return self.H_h, self.H_w
        device = self.H_h.device
        return _cached_hadamard(h_pad, device), _cached_hadamard(w_pad, device)

    @staticmethod
    def _adapt_param(param, target_h, target_w):
        """Bilinearly resample a 2-D parameter to (target_h, target_w)."""
        return F.interpolate(
            param.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(0)

    # ------------------------------------------------------------------

    def forward(self, x):
        height, width = x.shape[-2:]

        # Pad to next power of 2 (required by Hadamard)
        h_pad = find_min_power(height)
        w_pad = find_min_power(width)

        f0 = x
        if h_pad > height or w_pad > width:
            f0 = F.pad(f0, (0, w_pad - width, 0, h_pad - height))

        # ── 2-D forward WHT via two matmuls ────────────────────────────
        # H_h @ f0 @ H_w   (replaces 4x butterfly-loop calls)
        H_h, H_w = self._get_hadamard_pair(h_pad, w_pad)
        f2 = torch.matmul(H_h, torch.matmul(f0, H_w))

        # ── spectral processing (per pod) ──────────────────────────────
        need_adapt = (h_pad != self.height_pad or w_pad != self.width_pad)

        if self.pods == 1:
            # Fast path: avoid list / stack / sum overhead
            v_0 = (self._adapt_param(self.v[0], h_pad, w_pad)
                    if need_adapt else self.v[0])
            f3 = v_0 * f2
            f4 = self.conv[0](f3)
            if need_adapt:
                T_0 = self._adapt_param(self.ST[0].T, h_pad, w_pad)
                f6 = torch.copysign(
                    F.relu(torch.abs(f4) - F.relu(T_0)), f4)
            else:
                f6 = self.ST[0](f4)
        else:
            f5_list = []
            for i in range(self.pods):
                v_i = (self._adapt_param(self.v[i], h_pad, w_pad)
                        if need_adapt else self.v[i])
                f3 = v_i * f2
                f4 = self.conv[i](f3)
                if need_adapt:
                    T_i = self._adapt_param(self.ST[i].T, h_pad, w_pad)
                    f5 = torch.copysign(
                        F.relu(torch.abs(f4) - F.relu(T_i)), f4)
                else:
                    f5 = self.ST[i](f4)
                f5_list.append(f5)
            f6 = torch.stack(f5_list, dim=-1).sum(dim=-1)

        # ── 2-D inverse WHT via two matmuls ────────────────────────────
        # (H_h @ f6 @ H_w) / (h_pad * w_pad)
        y = torch.matmul(H_h, torch.matmul(f6, H_w))
        y = y * (1.0 / (h_pad * w_pad))       # multiply is faster than div

        # Crop to original spatial size
        y = y[..., :height, :width]

        if self.residual:
            y = y + x
        return y
