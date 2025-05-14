from model import common

import torch.nn as nn

#EDSRbaselince:n_resBlock=16 n_f=64    final:n_resBlock=32 n_f=256
#train_data:DIV2K800 DA:90,180,270；batch=16 patchs=48*48；ADAM 0.9 0.999 1e-8 ；lr=1e-4

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):#返回EDSR类的一个对象（就是edsr模型）
    return EDSR(args)

class EDSR(nn.Module):#（b：16；f：64；res_scale=1(不使用)）
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks#16baseline
        n_feats = args.n_feats#64
        kernel_size = 3 
        scale = args.scale[0]#2倍
        act = nn.ReLU(True)
        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)#加载已经训练好的EDSRpt参数文件
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)#对输入图片减DIV2K数据集的平均RGB像素值作预处理
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)#网络最后输出结果加回减去的平均像素值

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]#输入RGB，输出64个特征图

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)#循环16次
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

