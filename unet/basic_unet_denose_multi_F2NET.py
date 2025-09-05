# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union
import math
import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep
from functools import partial
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import DropPath
import torch.nn.functional as F

__all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]

class SpatialAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock3d, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.judge = nn.Conv3d(in_channels, in_channels // 8, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxZ )
        :return: affinity value + x
        B: batch size
        C: channels
        H: height
        W: width
        D: slice number (depth)
        """
        B, C, H, W, D = x.size()
        # compress x: [B,C,H,W,Z]-->[B,H*W*Z,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,W*H*D,C]
        proj_key = self.key(x).view(B, -1, W * H * D)  # -> [B,C,H*W*D]
        proj_judge = self.judge(x).view(B, -1, W * H * D).permute(0, 2, 1)  # -> [B,H*W*D,C]

        affinity1 = torch.matmul(proj_query, proj_key)
        affinity2 = torch.matmul(proj_judge, proj_key)
        affinity = torch.matmul(affinity1, affinity2)
        affinity = self.softmax(affinity)

        proj_value = self.value(x).view(B, -1, H * W * D)  # -> C*N
        weights = torch.matmul(proj_value, affinity)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock3d(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock3d, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxWxD )
        :return: affinity value + x
        """
        B, C, H, W, D = x.size()
        proj_query = x.view(B, C, -1).permute(0, 2, 1)
        proj_key = x.view(B, C, -1)
        proj_judge = x.view(B, C, -1).permute(0, 2, 1)
        affinity1 = torch.matmul(proj_key, proj_query)
        affinity2 = torch.matmul(proj_key, proj_judge)
        affinity = torch.matmul(affinity1, affinity2)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W, D)
        out = self.gamma * weights + x
        return out

class AffinityAttention3d(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention3d, self).__init__()
        self.sab = SpatialAttentionBlock3d(in_channels)
        self.cab = ChannelAttentionBlock3d(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab + x
        return out


class SeqConv3x3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1x1-conv3x3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = nn.Conv3d(self.inp_planes, self.mid_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = nn.Conv3d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1x1-sobelx':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 1, 0] = -2.0
                self.mask[i, 0, 0, 2, 0] = -1.0
                self.mask[i, 0, 0, 0, 2] = 1.0
                self.mask[i, 0, 0, 1, 2] = 2.0
                self.mask[i, 0, 0, 2, 2] = 1.0

                self.mask[i, 0, 1, 0, 0] = -2.0
                self.mask[i, 0, 1, 1, 0] = -4.0
                self.mask[i, 0, 1, 2, 0] = -2.0
                self.mask[i, 0, 1, 0, 2] = 2.0
                self.mask[i, 0, 1, 1, 2] = 4.0
                self.mask[i, 0, 1, 2, 2] = 2.0

                self.mask[i, 0, 2, 0, 0] = -1.0
                self.mask[i, 0, 2, 1, 0] = -2.0
                self.mask[i, 0, 2, 2, 0] = -1.0
                self.mask[i, 0, 2, 0, 2] = 1.0
                self.mask[i, 0, 2, 1, 2] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0

            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1x1-sobely':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 0, 1] = -2.0
                self.mask[i, 0, 0, 0, 2] = -1.0
                self.mask[i, 0, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2, 1] = 2.0
                self.mask[i, 0, 0, 2, 2] = 1.0

                self.mask[i, 0, 1, 0, 0] = -2.0
                self.mask[i, 0, 1, 0, 1] = -4.0
                self.mask[i, 0, 1, 0, 2] = -2.0
                self.mask[i, 0, 1, 2, 0] = 2.0
                self.mask[i, 0, 1, 2, 1] = 4.0
                self.mask[i, 0, 1, 2, 2] = 2.0

                self.mask[i, 0, 2, 0, 0] = -1.0
                self.mask[i, 0, 2, 0, 1] = -2.0
                self.mask[i, 0, 2, 0, 2] = -1.0
                self.mask[i, 0, 2, 2, 0] = 1.0
                self.mask[i, 0, 2, 2, 1] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0

            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1x1-sobelz':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 0, 1] = -2.0
                self.mask[i, 0, 0, 0, 2] = -1.0
                self.mask[i, 0, 0, 1, 0] = -2.0
                self.mask[i, 0, 0, 1, 1] = -4.0
                self.mask[i, 0, 0, 1, 2] = -2.0
                self.mask[i, 0, 0, 2, 0] = -1.0
                self.mask[i, 0, 0, 2, 1] = -2.0
                self.mask[i, 0, 0, 2, 2] = -1.0

                self.mask[i, 0, 2, 0, 0] = 1.0
                self.mask[i, 0, 2, 0, 1] = 2.0
                self.mask[i, 0, 2, 0, 2] = 1.0
                self.mask[i, 0, 2, 1, 0] = 2.0
                self.mask[i, 0, 2, 1, 1] = 4.0
                self.mask[i, 0, 2, 1, 2] = 2.0
                self.mask[i, 0, 2, 2, 0] = 1.0
                self.mask[i, 0, 2, 2, 1] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0

            self.mask = nn.Parameter(data=self.mask, requires_grad=False)


        elif self.type == 'conv1x1x1-laplacian':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1, 1] = 1.0

                self.mask[i, 0, 1, 1, 0] = 1.0
                self.mask[i, 0, 1, 0, 1] = 1.0
                self.mask[i, 0, 1, 1, 1] = -6.0
                self.mask[i, 0, 1, 2, 1] = 1.0
                self.mask[i, 0, 1, 1, 2] = 1.0

                self.mask[i, 0, 2, 1, 1] = 1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1x1-conv3x3x3':
            # conv-1x1x1
            y0 = F.conv3d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1, 1)  # B C H W D
            y0[:, :, 0:1, :, :] = b0_pad
            y0[:, :, -1:, :, :] = b0_pad
            y0[:, :, :, 0:1, :] = b0_pad
            y0[:, :, :, -1:, :] = b0_pad
            y0[:, :, :, :, 0:1] = b0_pad
            y0[:, :, :, :, -1:] = b0_pad
            # conv-3x3x3
            y1 = F.conv3d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv3d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1, 1)  # B C H W D
            y0[:, :, 0:1, :, :] = b0_pad
            y0[:, :, -1:, :, :] = b0_pad
            y0[:, :, :, 0:1, :] = b0_pad
            y0[:, :, :, -1:, :] = b0_pad
            y0[:, :, :, :, 0:1] = b0_pad
            y0[:, :, :, :, -1:] = b0_pad
            # conv-3x3x3
            y1 = F.conv3d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1x1-conv3x3x3':
            # re-param conv kernel
            RK = F.conv3d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3, 4))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, 3, device=device) * self.b0.view(1, -1, 1, 1, 1)
            RB = F.conv3d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :, :] = tmp[i, 0, :, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv3d(input=k1, weight=self.k0.permute(1, 0, 2, 3, 4))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, 3, device=device) * self.b0.view(1, -1, 1, 1, 1)
            RB = F.conv3d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class EoS(nn.Module):
    def __init__(self, dim, depth_multiplier=2, act_type='linear', with_idt=False, permute=True, **kwargs):
        super(EoS, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = dim
        self.out_planes = dim
        self.act_type = act_type
        self.permute = permute
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3x3 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1x1_sbx = SeqConv3x3x3('conv1x1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sby = SeqConv3x3x3('conv1x1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sbz = SeqConv3x3x3('conv1x1x1-sobelz', self.inp_planes, self.out_planes, -1)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 4, 1, 2, 3)  # (B, H, W, D, C) -> (B, C, H, W, D)

        if self.training:
            y = self.conv3x3x3(x) + \
                self.conv1x1x1_sbx(x) + \
                self.conv1x1x1_sby(x) + \
                self.conv1x1x1_sbz(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv3d(input=x, weight=RK, bias=RB, stride=1, padding=1)

        # y = self.conv3x3x3(x)     + \
        #     self.conv1x1x1_sbx(x) + \
        #     self.conv1x1x1_sby(x) + \
        #     self.conv1x1x1_sbz(x)
        # if self.with_idt:
        #     y += x

        if self.permute:
            y = y.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
            y = self.norm(y)

        if self.act_type != 'linear':
            y = y.permute(0, 4, 1, 2, 3)  # (B, H, W, D, C) -> (B, C, H, W, D)
            y = self.act(y)
            y = y.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3x3.weight, self.conv3x3x3.bias
        K1, B1 = self.conv1x1x1_sbx.rep_params()
        K2, B2 = self.conv1x1x1_sby.rep_params()
        K3, B3 = self.conv1x1x1_sbz.rep_params()
        RK, RB = (K0 + K1 + K2 + K3), (B0 + B1 + B2 + B3)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class EoL(nn.Module):
    def __init__(self, dim, depth_multiplier=2, act_type='linear', with_idt=False, permute=True, **kwargs):
        super(EoL, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = dim
        self.out_planes = dim
        self.act_type = act_type
        self.permute = permute
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3x3 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1x1_lpl = SeqConv3x3x3('conv1x1x1-laplacian', self.inp_planes, self.out_planes, -1)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 4, 1, 2, 3)  # (B, H, W, D, C) -> (B, C, H, W, D)

        if self.training:
            y = self.conv3x3x3(x) + \
                self.conv1x1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv3d(input=x, weight=RK, bias=RB, stride=1, padding=1)

        # y = self.conv3x3x3(x)     + \
        #     self.conv1x1x1_lpl(x)
        # if self.with_idt:
        #     y += x

        if self.permute:
            y = y.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
            y = self.norm(y)

        if self.act_type != 'linear':
            y = y.permute(0, 4, 1, 2, 3)  # (B, H, W, D, C) -> (B, C, H, W, D)
            y = self.act(y)
            y = y.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3x3.weight, self.conv3x3x3.bias
        K1, B1 = self.conv1x1x1_lpl.rep_params()
        RK, RB = (K0 + K1), (B0 + B1)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class Block(nn.Module):
    def __init__(self, dim,
                 token_mixer=nn.Identity,
                 mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None,
                 scale_trainable=True,
                 position_embedding=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop, act_type='prelu', with_idt=True, head_count=dim // 32)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value, trainable=scale_trainable) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value, trainable=scale_trainable) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.position_embedding = position_embedding

    def forward(self, x):
        if self.position_embedding:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(
                        self.token_mixer(self.norm1(self.position_embedding(x)))
                    )
                )
        else:
            x = self.res_scale1(x) + \
                self.layer_scale1(
                    self.drop_path1(
                        self.token_mixer(self.norm1(x))
                    )
                )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim, head_count=1, **kwargs):
        super().__init__()
        self.in_channels = dim
        self.key_channels = dim
        self.head_count = head_count
        self.value_channels = dim

        self.keys = nn.Conv3d(self.in_channels, self.key_channels, 1)
        self.queries = nn.Conv3d(self.in_channels, self.key_channels, 1)
        self.values = nn.Conv3d(self.in_channels, self.value_channels, 1)
        self.reprojection = nn.Conv3d(self.value_channels, self.in_channels, 1)

    def forward(self, input_):  # input : [B, H, W, D, C]
        input_ = input_.permute(0, 4, 1, 2, 3)  # [B, H, W, D, C] --> [B, C, H, W, D]
        n, _, h, w, d = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w * d))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w * d)
        values = self.values(input_).reshape((n, self.value_channels, h * w * d))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w, d)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        attention = attention.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
        return attention


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=3, padding=1,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv3d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self,
                x):  # fc(dim --> dim*expansion_ratio) --> act1 --> DWConv --> act2 --> fc(dim*expansion_ratio --> dim)
        x = self.pwconv1(x)  # input shape: [B, H, W, D, C]
        x = self.act1(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, H, W, D, C] --> [B, C, H, W, D]
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, H, W, D] --> [B, H, W, D, C]
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True,
                 # normalized_dim = -1 代表对最后一个维度即 channel, 每一个token做归一化，使得数据均值为0方差为1
                 bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        self.temb_proj = torch.nn.Linear(512,
                                         out_chns)

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

    def forward(self, x, temb):
        x = self.conv_0(x)
        x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        x = self.conv_1(x)
        return x


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

    def forward(self, x, temb):
        x = self.max_pooling(x)
        x = self.convs(x, temb)
        return x


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            bias: bool,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pre_conv: Optional[Union[nn.Module, str]] = "default",
            interp_mode: str = "linear",
            align_corners: Optional[bool] = True,
            halves: bool = True,
            dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor], temb):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1), temb)  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0, temb)

        return x


class EnhancedFeature(nn.Module):
    def __init__(self, in_chan, is_first=False):
        super(EnhancedFeature, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_chan),
            nn.ReLU(inplace=True),
        )


    def forward(self, x0, x1, x2, x3):
        w = torch.sigmoid(self.conv1(torch.cat((x1, x2, x3), dim=1)))
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        feat_x3 = torch.mul(x3, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2 + feat_x3), x1, x2, x3), dim=1))
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x + x1 + x2 + x3


class BasicUNetDe(nn.Module):
    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            dimensions: Optional[int] = None,
            norm_layers=partial(LayerNormGeneral, eps=1e-6, bias=False),
            mlps=Mlp,
            layer_scale_init_values=None,
            res_scale_init_values=[None, None, 1.0, 1.0],
            position_embeddings=[[None, None], [None, None], [None, None], [None, None]]
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(128,
                            512),
            torch.nn.Linear(512,
                            512),
        ])

        self.conv_0_dwi = TwoConv(spatial_dims, in_channels - 2, features[0], act, norm, bias, dropout)
        self.down_1_dwi = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_dwi = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_dwi = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4_dwi = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5_dwi = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.conv_0_adc = TwoConv(spatial_dims, in_channels - 2, features[0], act, norm, bias, dropout)
        self.down_1_adc = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_adc = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_adc = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4_adc = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5_adc = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.conv_0_flair = TwoConv(spatial_dims, in_channels - 2, features[0], act, norm, bias, dropout)
        self.down_1_flair = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_flair = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_flair = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4_flair = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5_flair = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.conv_0_fusion = TwoConv(spatial_dims, in_channels - 2, features[0], act, norm, bias, dropout)
        self.down_1_fusion = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2_fusion = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3_fusion = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4_fusion = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5_fusion = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.upcat_5 = UpCat(spatial_dims, fea[5] * 4, fea[4] * 4, fea[4] * 4, act, norm, bias, dropout, upsample)
        self.upcat_4 = UpCat(spatial_dims, fea[4] * 4, fea[3] * 4, fea[3] * 4, act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3] * 4, fea[2] * 4, fea[2] * 4, act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2] * 4, fea[1] * 4, fea[1] * 4, act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1] * 4, fea[0] * 4, fea[0] * 4, act, norm, bias, dropout, upsample,
                             halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[0] * 4, out_channels, kernel_size=1)

        self.fusion1 = EnhancedFeature(fea[0], is_first=True)
        self.fusion2 = EnhancedFeature(fea[1], is_first=False)
        self.fusion3 = EnhancedFeature(fea[2], is_first=False)
        self.fusion4 = EnhancedFeature(fea[3], is_first=False)
        self.fusion5 = EnhancedFeature(fea[4], is_first=False)

    def forward(self, x: torch.Tensor, t, embeddings=None, image=None, train=False):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        temb = get_timestep_embedding(t, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        if image is not None:
            dwi = image[:, 0, :].resize(image.shape[0], 1, image.shape[2], image.shape[3], image.shape[4])
            adc = image[:, 1, :].resize(image.shape[0], 1, image.shape[2], image.shape[3], image.shape[4])
            flair = image[:, 2, :].resize(image.shape[0], 1, image.shape[2], image.shape[3], image.shape[4])
            dwi = torch.cat([dwi, x], dim=1)
            adc = torch.cat([adc, x], dim=1)
            flair = torch.cat([flair, x], dim=1)

        x0_dwi = self.conv_0_dwi(dwi, temb)
        x0_adc = self.conv_0_adc(adc, temb)
        x0_flair = self.conv_0_flair(flair, temb)
        if embeddings is not None:
            x0_dwi += embeddings[0]
            x0_adc += embeddings[5]
            x0_flair += embeddings[10]

        x1_dwi = self.down_1_dwi(x0_dwi, temb)
        x1_adc = self.down_1_adc(x0_adc, temb)
        x1_flair = self.down_1_flair(x0_flair, temb)
        if embeddings is not None:
            x1_dwi += embeddings[1]
            x1_adc += embeddings[6]
            x1_flair += embeddings[11]

        x2_dwi = self.down_2_dwi(x1_dwi, temb)
        x2_adc = self.down_2_adc(x1_adc, temb)
        x2_flair = self.down_2_flair(x1_flair, temb)
        if embeddings is not None:
            x2_dwi += embeddings[2]
            x2_adc += embeddings[7]
            x2_flair += embeddings[12]

        x3_dwi = self.down_3_dwi(x2_dwi, temb)
        x3_adc = self.down_3_adc(x2_adc, temb)
        x3_flair = self.down_3_flair(x2_flair, temb)
        if embeddings is not None:
            x3_dwi += embeddings[3]
            x3_adc += embeddings[8]
            x3_flair += embeddings[13]

        x4_dwi = self.down_4_dwi(x3_dwi, temb)
        x4_adc = self.down_4_adc(x3_adc, temb)
        x4_flair = self.down_4_flair(x3_flair, temb)
        if embeddings is not None:
            x4_dwi += embeddings[4]
            x4_adc += embeddings[9]
            x4_flair += embeddings[14]

        x0_fusion = self.fusion1(0, x0_dwi, x0_adc, x0_flair)
        x1_fusion = self.down_1_fusion(x0_fusion, temb)
        x1_fusion = self.fusion2(x1_fusion, x1_dwi, x1_adc, x1_flair)
        x2_fusion = self.down_2_fusion(x1_fusion, temb)
        x2_fusion = self.fusion3(x2_fusion, x2_dwi, x2_adc, x2_flair)
        x3_fusion = self.down_3_fusion(x2_fusion, temb)
        x3_fusion = self.fusion4(x3_fusion, x3_dwi, x3_adc, x3_flair)
        x4_fusion = self.down_4_fusion(x3_fusion, temb)

        x4_concat = torch.cat((x4_fusion, x4_dwi, x4_adc, x4_flair), dim=1)
        x3_concat = torch.cat((x3_fusion, x3_dwi, x3_adc, x3_flair), dim=1)
        x2_concat = torch.cat((x2_fusion, x2_dwi, x2_adc, x2_flair), dim=1)
        x1_concat = torch.cat((x1_fusion, x1_dwi, x1_adc, x1_flair), dim=1)
        x0_concat = torch.cat((x0_fusion, x0_dwi, x0_adc, x0_flair), dim=1)

        u4 = self.upcat_4(x4_concat, x3_concat, temb)
        u3 = self.upcat_3(u4, x2_concat, temb)
        u2 = self.upcat_2(u3, x1_concat, temb)
        u1 = self.upcat_1(u2, x0_concat, temb)

        logits = self.final_conv(u1)
        # if train:
        #     # return [logits, embeddings[-1]+embeddings[-2]+embeddings[-3]]
        #     return logits
        # else:
        #     return logits
        return logits + embeddings[-1] + embeddings[-2] + embeddings[-3]



