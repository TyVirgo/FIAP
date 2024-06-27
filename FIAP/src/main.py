import os

import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import sys
sys.path.append("..//")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def net_par(net):
    num_par = 0
    for param in net.parameters():
        num_par += param.numel()
    print('Total number of parameters: %d' %num_par)

def print_setting(net, args):
    print('training model:', args.model)
    net_par(net)
    print('scale factor:', args.scale)
    print('resume from ', args.resume)
    print('output patch size', args.patch_size)
    print('optimization setting: ', args.optimizer)
    print('total epochs:', args.epochs, '    lr:', args.lr)
    print('save_name:', args.save)
    print('decay:', args.decay)


def main():
    global model
    if args.data_test == ['video']:#测试数据集名称包含“video”才执行if语句，option修改
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:#创建data对象、model对象、损失函数对象和训练对象（同级目录不用目录名.调用）
         # 会执行他们各自初始化函数依据option初始化部分值
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            print_setting(_model, args)
            while not t.terminate():#当前eopch小于设置的eopch返回false（网络未达到规定的epoch则继续训练和测试）
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
