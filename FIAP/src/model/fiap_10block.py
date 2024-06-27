import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../..//")
from src.model import common
from src.model import atten
from src.option import args

""""
    Args:输入输出特征图数量、放大尺度因子、输入输出通道不变3*3Conv
"""

def make_model(args, parent=False):
    return MYMODEL(args)

class high(nn.Module):
    def __init__(self, in_channel, scale):
        super(high, self).__init__()
        self.scale = scale

        self.up_down = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),

            nn.AvgPool2d(args.scale[0], stride=args.scale[0], padding=0),
        )

        self.sca = atten.SCA(in_channel)
        self.pa = atten.PA(in_channel)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.Sigmoid()
        )
        self.conv_3 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
        self.conv_3_last = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        out = self.conv_3(x)
        up = F.interpolate(x, scale_factor=self.scale[0], mode='bicubic', align_corners=False)

        high_info = x - self.up_down(up)

        atten_add = self.pa(high_info) + self.sca(high_info)
        atten = self.conv_1(atten_add)
        high_out = self.conv_3_last(torch.mul(atten, out))

        return high_out


class low(nn.Module):
    def __init__(self, in_channel):
        super(low, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, groups=in_channel)
        )

    def forward(self, x):
        return self.fc(x)

class block(nn.Module):
    def __init__(self, in_channel, scale):
        super(block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, in_channel // 2, 1)
        self.conv_2 = nn.Conv2d(in_channel, in_channel // 2, 1)

        self.high = high(in_channel // 2, scale)
        self.low = low(in_channel // 2)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            atten.ESA(in_channel, nn.Conv2d)
        )

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(x)

        out1 = self.high(out1)
        out2 = self.low(out2)

        out = torch.cat([out1, out2], dim=1)
        out = self.fc(out) + x

        return out

#####Net
class MYMODEL(nn.Module):
    def __init__(self, args):
        super(MYMODEL, self).__init__()

        feat = args.n_feats
        scale = args.scale
        self.scale = args.scale
        unf = 24

        self.head = nn.Sequential(nn.Conv2d(3, feat, 3, stride=1, padding=1), nn.PReLU())

        self.block1 = block(in_channel=feat, scale=scale)
        self.block2 = block(in_channel=feat, scale=scale)
        self.block3 = block(in_channel=feat, scale=scale)
        self.block4 = block(in_channel=feat, scale=scale)
        self.block5 = block(in_channel=feat, scale=scale)
        self.block6 = block(in_channel=feat, scale=scale)
        self.block7 = block(in_channel=feat, scale=scale)
        self.block8 = block(in_channel=feat, scale=scale)
        self.block9 = block(in_channel=feat, scale=scale)
        self.block10 = block(in_channel=feat, scale=scale)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_8 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
        )
        self.conv_9 = nn.Sequential(
            nn.Conv2d(feat * 2, feat, 1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, 3, 1, 1)
        )

        #### upsampling  UPA
        self.upconv1 = nn.Conv2d(feat, unf, 3, 1, 1, bias=True)
        self.att1 = atten.PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale[0] == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = atten.PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ################UPA

        #### upsampling  sub_pixel
        # self.sub_pixel = nn.Sequential(
        #     nn.Conv2d(feat, 3*((scale[0])**2), 3, 1, 1),
        #     nn.PixelShuffle((scale[0])),
        #     nn.PReLU()
        # )
        ############### sub_pixel

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)

        out = self.head(x)
        residual = out

        ### 10block
        out = self.block1(out)
        out1 = out
        out = self.block2(out)
        out2 = out
        out = self.conv_1(torch.cat([out, residual], dim=1))
        out = self.block3(out)
        out3 = out
        out = self.conv_2(torch.cat([out, out1], dim=1))
        out = self.block4(out)
        out4 = out
        out = self.conv_3(torch.cat([out, out2], dim=1))
        out = self.block5(out)
        out5 = out
        out = self.conv_4(torch.cat([out, out3], dim=1))
        out = self.block6(out)
        out6 = out
        out = self.conv_5(torch.cat([out, out4], dim=1))
        out = self.block7(out)
        out7 = out
        out = self.conv_6(torch.cat([out, out5], dim=1))
        out = self.block8(out)
        out8 = out
        out = self.conv_7(torch.cat([out, out6], dim=1))
        out = self.block9(out)
        out = self.conv_8(torch.cat([out, out7], dim=1))
        out = self.block10(out)
        out = self.conv_9(torch.cat([out, out8], dim=1))
        out = out + residual
        ###

        ####UPA
        if self.scale[0] == 2 or self.scale[0] == 3:
            out = self.upconv1(F.interpolate(out, scale_factor=self.scale[0], mode='nearest'))
            out = self.lrelu(self.att1(out))
            out = self.lrelu(self.HRconv1(out))
        elif self.scale[0] == 4:
            out = self.upconv1(F.interpolate(out, scale_factor=2, mode='nearest'))
            out = self.lrelu(self.att1(out))
            out = self.lrelu(self.HRconv1(out))
            out = self.upconv2(F.interpolate(out, scale_factor=2, mode='nearest'))
            out = self.lrelu(self.att2(out))
            out = self.lrelu(self.HRconv2(out))

        out = self.conv_last(out)
        ILR = F.interpolate(x, scale_factor=self.scale[0], mode='bilinear', align_corners=False)
        out = out + ILR

        # # ####sub_pixel upsample
        #out = self.sub_pixel(out)

        out = self.add_mean(out)
        return out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

if __name__ == '__main__':
    from thop import profile

    model = MYMODEL(args)
    scale = 2
    input = torch.randn(1, 3, 1280//scale, 720//scale)
    macs, params = profile(model, inputs=(input,))

    print('params=', params)
    print("MACs=", str(macs / 1e9) + '{}'.format("G"))
    print("MACs=", str(macs / 1e6) + '{}'.format("M"))