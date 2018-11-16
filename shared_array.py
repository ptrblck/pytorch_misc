"""
Script to demonstrate the usage of shared arrays using multiple workers.

In the first epoch the shared arrays in the dataset will be filled with
random values. After setting set_use_cache(True), the shared values will be
loaded from multiple processes.

@author: ptrblck
"""

import torch
from torch.utils.data import Dataset, DataLoader

import ctypes
import multiprocessing as mp

import numpy as np


class MyDataset(Dataset):
    def __init__(self):
        shared_array_base = mp.Array(ctypes.c_float, nb_samples*c*h*w)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(nb_samples, c, h, w)
        self.shared_array = torch.from_numpy(shared_array)
        self.use_cache = False

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        if not self.use_cache:
            print('Filling cache for index {}'.format(index))
            # Add your loading logic here
            self.shared_array[index] = torch.randn(c, h, w)
        x = self.shared_array[index]
        return x

    def __len__(self):
        return nb_samples


nb_samples, c, h, w = 10, 3, 24, 24

dataset = MyDataset()
loader = DataLoader(
    dataset,
    num_workers=2,
    shuffle=False
)

for epoch in range(2):
    for idx, data in enumerate(loader):
        print('Epoch {}, idx {}, data.shape {}'.format(epoch, idx, data.shape))

    if epoch == 0:
        loader.dataset.set_use_cache(True)
