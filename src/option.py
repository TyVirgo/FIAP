import datetime
import sys
sys.path.append("..//")
import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',#（model-init-）
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,#（model-init-  loss-init-）
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../dataset',#存储数据文件夹（修改）
                    help='dataset directory')#根目录是计算机/ /home/tyh123456/dataset
parser.add_argument('--dir_demo', type=str, default='../test',#自己测试图片的路径
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',#训练数据集文件夹名称（修改-data.init.py）DIV2K+FLICKR2K
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',#测试数据集文件夹名称（修改-data.init.py）
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800',#选一个数据范围作：训练集/测试集（修改-div2k.py）2650
                    help='train/test data range')#1-800/801-810：1-800作为训练集；801-810作为测试集
parser.add_argument('--ext', type=str, default='sep',#first convert .png to .npy. for all the training images (.png)
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',#超分辨率放大因子（修改-model_init_ trainer.py）
                    help='super resolution scale')#创建模型初始化时设置
parser.add_argument('--patch_size', type=int, default=128,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,#输入图像的特征通道数量
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true', #（model-init-）
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='MYEDSR',#要使用模型的名称（修改-model—init-）
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')#../pre_models/edsr_baseline_x2-1bc95232.pt
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,#RCAN要设置20个
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=32,#RCAN64
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',#模型计算使用的精度（model-init-）
                    choices=('single', 'half'),#半浮点half：单浮点single
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,#10个残差组（rcan.py-RCAN-）
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,#通道注意力缩放通道的倍数
                    help='number of feature maps reduction')

# Option for FMEN
parser.add_argument('--down_blocks', type=int, default=4,
                    help='number of pair of ERB and HFAB')
parser.add_argument('--up_blocks', type=str, default='2+1+1+1+1',
                    help='number of ERB in HFAB')
parser.add_argument('--mid_feats', type=int, default='16',
                    help='Number of feature maps in ERB')
parser.add_argument('--backbone_expand_ratio', type=int, default=2,
                    help='Expand ratio of RRRB in trunk ERB')
parser.add_argument('--attention_expand_ratio', type=int, default=2,
                    help='Expand ratio of RRRB in branch ERB')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,#每次测试用了多少张图片（batches修改）data
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true', #测试是否使用几何自集成（model-init-）
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', #是否只测试模型（div2k.py、trainer.py.terminate）
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=6e-4, #学习率 (utility.py.make_optimizer)
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200-400-600-800',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',#参数优化方案(utility.py.make_optimizer)
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,#权重衰退(utility.py.make_optimizer)
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',#损失函数，用+拼接使用多个（loss-init-）
                    help='loss function configuration')#格式：权重*损失函数类型
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Log specifications
parser.add_argument('--save', type=str, default=now_time,#保存训练结果的路径
                    help='file name to save')
parser.add_argument('--load', type=str, default='',#要加载的文件名称（loss-init-）
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',#（model-init-）#保存每个epoch训练模型pt文件
                    help='save all intermediate pre_models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',#保存每个测试集的SR图片
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',#保存每个测试集的LR图和HR图
                    help='save low-resolution and high-resolution images together')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')#切割字符串，返回列表
args.up_blocks = list(map(lambda x: int(x), args.up_blocks.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

