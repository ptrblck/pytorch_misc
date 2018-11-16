"""
Model sharding with DataParallel using 2 pairs of 2 GPUs.

@author: ptrblck
"""

import torch
import torch.nn as nn


class SubModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SubModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        print('SubModule, device: {}, shape: {}\n'.format(x.device, x.shape))
        x = self.conv1(x)
        return x


class MyModel(nn.Module):
    def __init__(self, split_gpus, parallel):
        super(MyModel, self).__init__()
        self.module1 = SubModule(3, 6)
        self.module2 = SubModule(6, 1)

        self.split_gpus = split_gpus
        self.parallel = parallel
        if self.split_gpus and self.parallel:
                self.module1 = nn.DataParallel(self.module1, device_ids=[0, 1]).to('cuda:0')
                self.module2 = nn.DataParallel(self.module2, device_ids=[2, 3]).to('cuda:2')

    def forward(self, x):
        print('Input: device {}, shape {}\n'.format(x.device, x.shape))
        x = self.module1(x)
        print('After module1: device {}, shape {}\n'.format(x.device, x.shape))
        x = self.module2(x)
        print('After module2: device {}, shape {}\n'.format(x.device, x.shape))
        return x


model = MyModel(split_gpus=True, parallel=True)
x = torch.randn(16, 3, 24, 24).to('cuda:0')
output = model(x)
