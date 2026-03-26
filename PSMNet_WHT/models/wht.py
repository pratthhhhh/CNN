"""
Optimized Walsh-Hadamard Transform convolution layer.

Backend priority (auto-selected):
  1. fast_hadamard_transform  — CUDA-native, pip install fast-hadamard-transform
  2. torch.compile'd butterfly — PyTorch >= 2.0, fuses kernel launches
  3. Plain Python butterfly     — universal fallback

Why the matmul approach was still slow:
  Matmul is O(n^2) per spatial dim. At test resolution (368x1232 -> padded
  to 256x1024) the right-multiply alone does 256*1024*1024 = 268M FLOPs per
  (B,C) slice. The butterfly is O(n log n) = ~10K ops per element — 100x fewer
  FLOPs at n=1024. The butterfly just needs its ~100 tiny kernel launches
  fused, which torch.compile handles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ─────────────────────────────────────────────────────────────────

def find_min_power(x, p=2):
    """Smallest power of *p* that is >= x."""
    y = 1
    while y < x:
        y *= p
    return y


# ── Backend selection ───────────────────────────────────────────────────────

_BACKEND = 'python'     # updated below

# --- Tier 1: CUDA-native (Tri Dao's fast-hadamard-transform) ---------------
try:
    from fast_hadamard_transform import hadamard_transform as _cuda_fwht

    def _fwht_last(x):
        return _cuda_fwht(x.contiguous())

    _BACKEND = 'cuda_native'

except ImportError:

    # --- Tier 2 / 3: Butterfly (compiled or plain) -------------------------
    def _fwht_last(x):
        """Butterfly FWHT along the LAST dimension.  O(n log n).

        No .clone() — every butterfly stage produces new tensors via
        stack(), so the input is never mutated.
        """
        n = x.shape[-1]
        h = 1
        while h < n:
            x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
            a = x[..., 0, :]
            b = x[..., 1, :]
            x = torch.stack([a + b, a - b], dim=-2)
            x = x.view(*x.shape[:-3], n)
            h *= 2
        return x

    if hasattr(torch, 'compile'):
        try:
            _fwht_last = torch.compile(_fwht_last)
            _BACKEND = 'compiled_butterfly'
        except Exception:
            _BACKEND = 'python'
    else:
        _BACKEND = 'python'


def fwht(x, axis=-1):
    """Forward Walsh-Hadamard Transform along *axis*."""
    if axis == -1 or axis == x.ndim - 1:
        return _fwht_last(x)
    x = x.transpose(axis, -1).contiguous()
    y = _fwht_last(x)
    return y.transpose(axis, -1).contiguous()


def ifwht(x, axis=-1):
    """Inverse Walsh-Hadamard Transform along *axis*."""
    n = x.shape[axis]
    return fwht(x, axis=axis) * (1.0 / n)


print(f'[WHT] backend = {_BACKEND}')


# ── Soft thresholding ───────────────────────────────────────────────────────

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
    """Walsh-Hadamard-Transform 2-D convolution.

    Constructor signature is identical to the original so that submodule.py
    needs no changes.
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

    # ------------------------------------------------------------------

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

        # Pad to next power of 2
        h_pad = find_min_power(height)
        w_pad = find_min_power(width)

        f0 = x
        if h_pad > height or w_pad > width:
            f0 = F.pad(f0, (0, w_pad - width, 0, h_pad - height))

        # ── 2-D forward WHT (2 compiled butterfly calls) ──────────────
        f2 = fwht(fwht(f0, axis=-1), axis=-2)

        # ── spectral processing ────────────────────────────────────────
        need_adapt = (h_pad != self.height_pad or w_pad != self.width_pad)

        if self.pods == 1:
            # Fast path — skip list/stack/sum overhead
            v_0 = (self._adapt_param(self.v[0], h_pad, w_pad)
                    if need_adapt else self.v[0])
            f4 = self.conv[0](v_0 * f2)
            if need_adapt:
                T_0 = self._adapt_param(self.ST[0].T, h_pad, w_pad)
                f6 = torch.copysign(
                    F.relu(torch.abs(f4) - F.relu(T_0)), f4)
            else:
                f6 = self.ST[0](f4)
        else:
            parts = []
            for i in range(self.pods):
                v_i = (self._adapt_param(self.v[i], h_pad, w_pad)
                       if need_adapt else self.v[i])
                f4 = self.conv[i](v_i * f2)
                if need_adapt:
                    T_i = self._adapt_param(self.ST[i].T, h_pad, w_pad)
                    f5 = torch.copysign(
                        F.relu(torch.abs(f4) - F.relu(T_i)), f4)
                else:
                    f5 = self.ST[i](f4)
                parts.append(f5)
            f6 = torch.stack(parts, dim=-1).sum(dim=-1)

        # ── 2-D inverse WHT ───────────────────────────────────────────
        y = ifwht(ifwht(f6, axis=-1), axis=-2)

        # Crop back to original spatial size
        y = y[..., :height, :width]

        if self.residual:
            y = y + x
        return y
