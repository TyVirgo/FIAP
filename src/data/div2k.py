import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        #先分割“/”得到['1-800','801-810']列表，r遍历该列表，每次遍历继续分割“-”得到两个列表,两次遍历得到列表[['1','800'],['801','810']]
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0] #data_range范围设置为['1','800']就是训练集DIV2K的前800张图片（用列表元素覆盖整个列表）
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))#将列表字符串转换为int表示开始为1，结束为800
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

