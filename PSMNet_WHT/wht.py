import numpy as np
import torch
from scipy.linalg import hadamard


def find_min_power(x, p=2):
    """Return the smallest power of p that is >= x."""
    y = 1
    while y < x:
        y *= p
    return y


# ---------------------------------------------------------------------------
# Soft Thresholding
# ---------------------------------------------------------------------------

class SoftThresholding(torch.nn.Module):
    """
    Per-channel soft thresholding.

    Threshold is learned per output channel (shape: [C, 1, 1]) rather than
    per spatial frequency bin.  This is more expressive for stereo feature
    maps where channels carry semantic content (edges, textures) and keeps
    the parameter count independent of input resolution.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.T = torch.nn.Parameter(torch.rand(num_channels, 1, 1) / 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.copysign(
            torch.nn.functional.relu(
                torch.abs(x) - torch.nn.functional.relu(self.T)
            ),
            x,
        )


# ---------------------------------------------------------------------------
# Walsh–Hadamard Transform (forward and inverse)
# ---------------------------------------------------------------------------

def fwht(u: torch.Tensor, axis: int = -1, fast: bool = True, normalize: bool = False) -> torch.Tensor:
    """
    Fast Walsh–Hadamard Transform along *axis*.

    Computes  H_n @ u  (or u @ H_n for the last axis) where H_n is the
    unnormalised Hadamard matrix of size n×n.  n must be a power of 2.

    Parameters
    ----------
    u         : Input tensor.
    axis      : Axis along which to apply the transform (default: last axis).
    fast      : Use the O(n log n) butterfly algorithm (default: True).
                Set False to use the explicit matrix multiply (debug only).
    normalize : If True, divide the output by n, turning this into the
                inverse WHT (ifwht).  Kept False by default (forward WHT).

    Returns
    -------
    Tensor of the same shape as *u*.
    """
    if axis != -1:
        u = torch.transpose(u, -1, axis)

    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, "n must be a power of 2"

    if fast:
        x = u[..., np.newaxis]
        for _ in range(m)[::-1]:
            x = torch.cat(
                (x[..., ::2, :] + x[..., 1::2, :],
                 x[..., ::2, :] - x[..., 1::2, :]),
                dim=-1,
            )
        y = x.squeeze(-2)
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H

    if normalize:
        y = y / n

    if axis != -1:
        y = torch.transpose(y, -1, axis)

    return y


def ifwht(u: torch.Tensor, axis: int = -1, fast: bool = True) -> torch.Tensor:
    """
    Inverse Fast Walsh–Hadamard Transform along *axis*.

    Identical butterfly to fwht but divides by n at the end, since
    H_n^{-1} = H_n / n.
    """
    return fwht(u, axis=axis, fast=fast, normalize=True)


# ---------------------------------------------------------------------------
# WHTConv2D
# ---------------------------------------------------------------------------

class WHTConv2D(torch.nn.Module):
    """
    2-D convolution in the Walsh–Hadamard domain.

    Improvements over the original implementation
    ----------------------------------------------
    1. **Log-space frequency filter** (``log_v``): the per-frequency-bin scale
       is stored as log(v) and exponentiated in the forward pass, keeping it
       strictly positive and giving smoother gradient flow into the backbone.

    2. **Batched pods**: instead of ``pods`` sequential Conv2d kernel launches,
       a single grouped Conv2d processes all pods in one CUDA call.

    3. **Per-channel SoftThresholding**: threshold is [out_channels, 1, 1]
       rather than [H_pad, W_pad], so it adapts to channel semantics rather
       than spatial frequency positions and stays resolution-independent.

    4. **Dynamic padding**: spatial size is inferred at runtime, removing the
       fixed (height, width) constructor requirement and the hard exception.
       The ``height`` / ``width`` constructor arguments are kept as *hints* for
       pre-computing pad sizes but are no longer enforced.

    Parameters
    ----------
    in_channels  : Number of input channels.
    out_channels : Number of output channels.
    pods         : Number of parallel WHT filter branches (default: 1).
                   More pods → higher capacity; set 2-4 to recover accuracy
                   when replacing 3×3 convs in the stereo backbone.
    residual     : Add input residual connection (default: True).
    height, width: Optional spatial size hints used only to pre-compute the
                   padded dimensions.  Pass 0 (default) for fully dynamic mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pods: int = 1,
        residual: bool = True,
        height: int = 0,
        width: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.residual = residual

        # Hint-based pre-computation (optional; overridden at runtime if 0)
        self._hint_h = height
        self._hint_w = width

        # Single grouped Conv2d replaces pods-length ModuleList:
        #   groups=pods  →  each pod's channels are processed independently,
        #   then the output is summed across pods in forward().
        self.conv = torch.nn.Conv2d(
            in_channels * pods,
            out_channels * pods,
            kernel_size=1,
            bias=False,
            groups=pods,
        )

        # Per-channel soft thresholding (one module shared across pods for
        # parameter efficiency; out_channels applies after pod summation).
        self.ST = SoftThresholding(out_channels)

        # Log-space frequency filter: one scalar per (pod, H_pad, W_pad).
        # Initialised to 0 so exp(log_v) = 1 at the start of training.
        # Registered as a ParameterList so each pod's filter is independent.
        # Size is set lazily on the first forward pass if hints are 0.
        if height > 0 and width > 0:
            h_pad = find_min_power(height)
            w_pad = find_min_power(width)
            self.log_v = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(h_pad, w_pad))
                for _ in range(pods)
            ])
        else:
            self.log_v = None  # initialised lazily in forward()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_log_v(self, h_pad: int, w_pad: int, device: torch.device):
        """Lazily initialise log_v when spatial size is first seen."""
        self.log_v = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(h_pad, w_pad, device=device))
            for _ in range(self.pods)
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # --- Dynamic padding to next power of 2 ---
        h_pad = find_min_power(H)
        w_pad = find_min_power(W)

        # Lazy log_v initialisation (first call or resolution change)
        if self.log_v is None or self.log_v[0].shape != (h_pad, w_pad):
            self._init_log_v(h_pad, w_pad, x.device)

        f0 = x
        if h_pad > H or w_pad > W:
            f0 = torch.nn.functional.pad(f0, (0, w_pad - W, 0, h_pad - H))

        # --- 2-D WHT (row then column) ---
        f1 = fwht(f0, axis=-1)   # along width
        f2 = fwht(f1, axis=-2)   # along height
        # f2: (B, C, h_pad, w_pad)

        # --- Frequency-domain filtering (batched across pods) ---
        # Stack log_v: (pods, h_pad, w_pad) → broadcast over (B, C)
        v_stack = torch.stack(
            [torch.exp(self.log_v[i]) for i in range(self.pods)], dim=0
        )  # (pods, h_pad, w_pad)

        # Expand f2 for each pod: (B, pods, C, h_pad, w_pad)
        f2_expanded = f2.unsqueeze(1) * v_stack.unsqueeze(0).unsqueeze(2)

        # Merge (B, pods, C, ...) → (B, pods*C, ...) for grouped conv
        f3 = f2_expanded.view(B, self.pods * C, h_pad, w_pad)

        # Single grouped conv launch: (B, pods*C, h_pad, w_pad) → (B, pods*out_C, ...)
        f4 = self.conv(f3)  # (B, pods * out_channels, h_pad, w_pad)

        # Sum across pods: (B, pods, out_channels, h_pad, w_pad) → (B, out_channels, ...)
        f5 = f4.view(B, self.pods, self.out_channels, h_pad, w_pad).sum(dim=1)

        # --- Per-channel soft thresholding ---
        f6 = self.ST(f5)  # (B, out_channels, h_pad, w_pad)

        # --- 2-D inverse WHT ---
        f7 = ifwht(f6, axis=-1)
        f8 = ifwht(f7, axis=-2)

        # Crop back to original spatial size
        y = f8[..., :H, :W]

        # Residual connection (only valid when in_channels == out_channels)
        if self.residual:
            if C == self.out_channels:
                y = y + x
            else:
                # Channel mismatch: skip residual silently rather than crash
                pass

        return y