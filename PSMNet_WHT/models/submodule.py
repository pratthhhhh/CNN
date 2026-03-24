from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wht import WHTConv2D


# ---------------------------------------------------------------------------
# Standard convolution helpers (unchanged)
# ---------------------------------------------------------------------------

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes,
                  kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes,
                  kernel_size=kernel_size, padding=pad,
                  stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


# ---------------------------------------------------------------------------
# WHT-based convolution helper
# ---------------------------------------------------------------------------

def convbn_wht(in_planes, out_planes, height, width):
    """Drop-in replacement for stride-1 convbn inside the feature extractor.

    Uses a single WHT pod (pods=1).  residual=False so the standard
    BasicBlock residual path (out += x) is not doubled.
    """
    return nn.Sequential(
        WHTConv2D(height, width, in_planes, out_planes, pods=1, residual=False),
        nn.BatchNorm2d(out_planes))


# ---------------------------------------------------------------------------
# BasicBlock – WHT-aware
# ---------------------------------------------------------------------------
#
# Replacement strategy:
#   • WHT replaces a 3×3 convbn when ALL of these hold:
#       – stride == 1  (WHTConv2D has no stride mechanism)
#       – dilation == 1 (WHTConv2D has no dilation mechanism)
#       – height & width are known at construction time
#   • Otherwise the original standard convbn is kept.
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation,
                 height=None, width=None):
        super(BasicBlock, self).__init__()

        use_wht = (height is not None and width is not None)

        # ── conv1 ──────────────────────────────────────────────────────────
        if use_wht and stride == 1 and dilation == 1:
            self.conv1 = nn.Sequential(
                convbn_wht(inplanes, planes, height, width),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                convbn(inplanes, planes, 3, stride, pad, dilation),
                nn.ReLU(inplace=True))

        # ── conv2 ──────────────────────────────────────────────────────────
        # conv2 is always stride-1. After a stride-2 conv1, spatial dims halve.
        if use_wht and dilation == 1:
            h2 = height // stride
            w2 = width  // stride
            self.conv2 = convbn_wht(planes, planes, h2, w2)
        else:
            self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out


# ---------------------------------------------------------------------------
# Disparity regression helper (unchanged)
# ---------------------------------------------------------------------------

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        device = x.device
        dtype  = x.dtype
        disp = torch.arange(self.maxdisp, device=device, dtype=dtype).view(1, self.maxdisp, 1, 1)
        out  = torch.sum(x * disp, 1, keepdim=True)
        return out


# ---------------------------------------------------------------------------
# Feature extraction backbone – WHT-integrated
# ---------------------------------------------------------------------------
#
# Spatial dimension tracking (default 256×512 input):
#   Input image      : 256 × 512
#   After firstconv  : 128 × 256  (stride-2 in first convbn)
#   After layer1     : 128 × 256  (stride-1)
#   After layer2     : 64  × 128  (stride-2 in first block)
#   After layer3     : 64  × 128  (stride-1)
#   After layer4     : 64  × 128  (stride-1, dilation=2 → falls back to std)
#   SPP branches     : pooled then upsampled back to 64 × 128
#   lastconv         : 64  × 128
#
# WHT replacements (bold = replaced):
#   firstconv  :  std-s2 → **WHT** → **WHT**
#   layer1     :  all blocks **WHT** (stride-1, dilation-1)
#   layer2     :  first block conv1 std-s2, conv2 **WHT**;
#                 remaining blocks **WHT** (stride-1)
#   layer3     :  all blocks **WHT** (stride-1, dilation-1)
#   layer4     :  std (dilation=2, WHT not applicable)
#   SPP        :  std (variable tiny spatial sizes after pooling)
#   lastconv   :  **WHT** (3×3 at 64×128) + std 1×1
# ---------------------------------------------------------------------------

class feature_extraction(nn.Module):
    def __init__(self, img_height=256, img_width=512):
        super(feature_extraction, self).__init__()
        self.inplanes = 32

        # Spatial dims after stride-2 first conv
        h1 = img_height // 2   # 128
        w1 = img_width  // 2   # 256

        # ── firstconv ──────────────────────────────────────────────────────
        self.firstconv = nn.Sequential(
            convbn(3, 32, 3, 2, 1, 1),              # stride-2, standard
            nn.ReLU(inplace=True),
            convbn_wht(32, 32, h1, w1),             # stride-1, WHT
            nn.ReLU(inplace=True),
            convbn_wht(32, 32, h1, w1),             # stride-1, WHT
            nn.ReLU(inplace=True))

        # ── residual layers ────────────────────────────────────────────────
        self.layer1 = self._make_layer(BasicBlock, 32,  3,  1, 1, 1,
                                       height=h1, width=w1)

        self.layer2 = self._make_layer(BasicBlock, 64,  16, 2, 1, 1,
                                       height=h1, width=w1)

        h2 = h1 // 2   # 64
        w2 = w1 // 2   # 128

        self.layer3 = self._make_layer(BasicBlock, 128, 3,  1, 1, 1,
                                       height=h2, width=w2)

        # dilation=2 → WHT falls back to standard conv inside BasicBlock
        self.layer4 = self._make_layer(BasicBlock, 128, 3,  1, 1, 2,
                                       height=h2, width=w2)

        # ── SPP branches (standard – variable tiny spatial sizes) ──────────
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=True))

        # ── lastconv ──────────────────────────────────────────────────────
        # Concatenated features are at h2×w2.  Replace the 3×3 conv with WHT.
        self.lastconv = nn.Sequential(
            convbn_wht(320, 128, h2, w2),           # WHT replaces 3×3 conv
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    # ─────────────────────────────────────────────────────────────────────

    def _make_layer(self, block, planes, blocks, stride, pad, dilation,
                    height=None, width=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation,
                            height=height, width=width))
        self.inplanes = planes * block.expansion

        h_next = height // stride if height is not None else None
        w_next = width  // stride if width  is not None else None

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation,
                                height=h_next, width=w_next))

        return nn.Sequential(*layers)

    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1,
                                       size=(output_skip.size()[2], output_skip.size()[3]),
                                       mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2,
                                       size=(output_skip.size()[2], output_skip.size()[3]),
                                       mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3,
                                       size=(output_skip.size()[2], output_skip.size()[3]),
                                       mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4,
                                       size=(output_skip.size()[2], output_skip.size()[3]),
                                       mode='bilinear', align_corners=False)

        output_feature = torch.cat((output_raw, output_skip,
                                    output_branch4, output_branch3,
                                    output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature