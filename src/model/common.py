import math

import torch
import torch.nn as nn
# from model import atten
# import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):#保持输入特征图和输出特征图数量不变
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        #self.res_scale = res_scale#残差缩放因子

    def forward(self, x):
        res = self.body(x)#.mul(self.res_scale)
        res += x

        return res

#残差分支和残差求和输出
class ResBlock_RFA(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_RFA, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        #self.res_scale = res_scale#残差缩放因子

    def forward(self, x):
        res = self.body(x)#.mul(self.res_scale)
        out = res + x

        return res, out

class RFA(nn.Module):#残差特征融合（）
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RFA, self).__init__()
        self.rb1 = ResBlock_RFA(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.rb2 = ResBlock_RFA(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.rb3 = ResBlock_RFA(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.rb4 = ResBlock_RFA(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        # self.rb1 = atten.SEResBlock(conv, n_feats, kernel_size, bias, act, res_scale)
        # self.rb2 = atten.SEResBlock(conv, n_feats, kernel_size, bias, act, res_scale)
        # self.rb3 = atten.SEResBlock(conv, n_feats, kernel_size, bias, act, res_scale)
        # self.rb4 = atten.SEResBlock(conv, n_feats, kernel_size, bias, act, res_scale)
        # self.rb1 = atten.GCTResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        # self.rb2 = atten.GCTResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        # self.rb3 = atten.GCTResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        # self.rb4 = atten.GCTResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.res_scale = res_scale#残差缩放因子
        self.conv1 = conv(n_feats*4, n_feats, 1, bias=bias)

    def forward(self, x):

        res1, out = self.rb1(x)
        res2, out = self.rb2(out)
        res3, out = self.rb3(out)
        temp = self.rb4(out)
        out = torch.cat([res1, res2, res3, temp[0]], dim=1)
        out = self.conv1(out) + x

        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?(左到右位运算)
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))#亚像素卷积
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


