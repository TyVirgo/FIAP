from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train: #遍历训练数据集（若干个数据集由+号构成一个名称）
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                #动态调用python包（m相当于data.div2k这个py）
                m = import_module('data.' + module_name.lower())#选择要使用模型的名称（自动转换为小写）option的--model修改
                #相当于append了一个m（data.div2k）的DIV2K类的一个实例对象
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader( #创建Dataloader类一个实例对象
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,#是否只使用CPU
                num_workers=args.n_threads,#加载数据使用的线程数量
            )

        self.loader_test = []
        for d in args.data_test:#设置测试数据集的（option修改）
            if d in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:#使用基准数据集进行测试（如果datat_test包含其中一个数据集名称）
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:#使用DIV2K作为测试集
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,#一张图片一张图片作为测试输入
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
