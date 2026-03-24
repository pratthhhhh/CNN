import numpy as np
import torch
from scipy.linalg import hadamard

def find_min_power(x, p=2):
    y = 1
    while y<x:
        y *= p
    return y

class SoftThresholding(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.T = torch.nn.Parameter(torch.rand(self.num_features)/10)
              
    def forward(self, x):
        return torch.copysign(torch.nn.functional.relu(torch.abs(x)-torch.nn.functional.relu(self.T)), x) 

def fwht(u, axis=-1, fast=True):
    """Fast Walsh-Hadamard Transform: multiply H_n @ u.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        axis: axis along which to apply the transform
        fast: if True uses butterfly O(n log n), else uses full matrix O(n^2)
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    if fast:
        x = u.clone()
        h = 1
        while h < n:
            x = x.view(*x.shape[:-1], n // (2*h), 2*h)
            a = x[..., :h]
            b = x[..., h:]
            x = torch.cat([a + b, a - b], dim=-1)
            x = x.view(*x.shape[:-2], n)
            h *= 2
        y = x
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H

    if axis != -1:
        y = torch.transpose(y, -1, axis)
    return y

def ifwht(u, axis=-1, fast=True):
    """Inverse Fast Walsh-Hadamard Transform: multiply H_n @ u / n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        axis: axis along which to apply the transform
        fast: if True uses butterfly O(n log n), else uses full matrix O(n^2)
    Returns:
        product: Tensor of shape (..., n)
    """  
    if axis != -1:
        u = torch.transpose(u, -1, axis)
    
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    if fast:
        x = u.clone()
        h = 1
        while h < n:
            x = x.view(*x.shape[:-1], n // (2*h), 2*h)
            a = x[..., :h]
            b = x[..., h:]
            x = torch.cat([a + b, a - b], dim=-1)
            x = x.view(*x.shape[:-2], n)
            h *= 2
        y = x / n
    else:
        H = torch.tensor(hadamard(n), dtype=torch.float, device=u.device)
        y = u @ H / n

    if axis != -1:
        y = torch.transpose(y, -1, axis)
        
    return y

class WHTConv2D(torch.nn.Module):
    def __init__(self, height, width, in_channels, out_channels, pods = 1, residual=True):
        super().__init__()
        self.height = height       
        self.width = width
        self.height_pad = find_min_power(self.height)  
        self.width_pad = find_min_power(self.width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pods = pods
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels, 1, bias=False) for i in range(self.pods)])
        self.ST = torch.nn.ModuleList([SoftThresholding((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.v = torch.nn.ParameterList([torch.rand((self.height_pad, self.width_pad)) for i in range(self.pods)])
        self.residual = residual
        
        
    def _adapt_param(self, param, target_h, target_w):
        """Bilinearly interpolate a 2-D learnable parameter to (target_h, target_w).

        During training the sizes match so this is never called (zero cost).
        At test time with a different image size the spectral-domain parameters
        are smoothly resampled to the new padded dimensions.
        """
        # param shape: (h, w) → (1, 1, h, w) for F.interpolate
        return torch.nn.functional.interpolate(
            param.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)

    def forward(self, x):
        height, width = x.shape[-2:]

        # Dynamic padding to next power of 2 for the *actual* input size
        h_pad = find_min_power(height)
        w_pad = find_min_power(width)

        f0 = x
        if h_pad > height or w_pad > width:
            f0 = torch.nn.functional.pad(f0, (0, w_pad - width, 0, h_pad - height))

        # Forward WHT along both spatial axes
        f1 = fwht(f0, axis=-1, fast=True)
        f2 = fwht(f1, axis=-2, fast=True)

        # Check whether we need to adapt spectral parameters
        need_adapt = (h_pad != self.height_pad or w_pad != self.width_pad)

        f5_list = []
        for i in range(self.pods):
            # Spectral weighting
            v_i = self._adapt_param(self.v[i], h_pad, w_pad) if need_adapt else self.v[i]
            f3 = v_i * f2

            # 1×1 channel mixing (spatially agnostic, no adaptation needed)
            f4 = self.conv[i](f3)

            # Soft thresholding
            if need_adapt:
                T_i = self._adapt_param(self.ST[i].T, h_pad, w_pad)
                f5 = torch.copysign(
                    torch.nn.functional.relu(torch.abs(f4) - torch.nn.functional.relu(T_i)),
                    f4)
            else:
                f5 = self.ST[i](f4)

            f5_list.append(f5)

        f6 = torch.stack(f5_list, dim=-1).sum(dim=-1)

        # Inverse WHT
        f7 = ifwht(f6, axis=-1, fast=True)
        f8 = ifwht(f7, axis=-2, fast=True)

        # Crop back to original spatial size
        y = f8[..., :height, :width]

        if self.residual:
            y = y + x
        return y
