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
        
        
    def forward(self, x):
        height, width = x.shape[-2:]
        if height!= self.height or width!=self.width:
            raise Exception('({}, {})!=({}, {})'.format(height, width, self.height, self.width))
     
        f0 = x
        if self.width_pad>self.width or self.height_pad>self.height:
            f0 = torch.nn.functional.pad(f0, (0, self.width_pad-self.width, 0, self.height_pad-self.height))
        
        f1 = fwht(f0, axis=-1, fast=True)
        f2 = fwht(f1, axis=-2, fast=True)
        
        f3 = [self.v[i]*f2 for i in range(self.pods)]
        f4 = [self.conv[i](f3[i]) for i in range(self.pods)]
        f5 = [self.ST[i](f4[i]) for i in range(self.pods)]
        
        f6 = torch.stack(f5, dim=-1).sum(dim=-1)
        
        f7 = ifwht(f6, axis=-1, fast=True)
        f8 = ifwht(f7, axis=-2, fast=True)
        
        y = f8[..., :self.height, :self.width]
        
        if self.residual:
            y = y + x
        return y