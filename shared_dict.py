"""
Script to demonstrate the usage of shared dicts using multiple workers.

In the first epoch the shared dict in the dataset will be filled with
random values. The next epochs will just use the dict without "loading" the
data again.

@author: ptrblck
"""

from multiprocessing import Manager

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, shared_dict, length):
        self.shared_dict = shared_dict
        self.length = length

    def __getitem__(self, index):
        if index not in self.shared_dict:
            print('Adding {} to shared_dict'.format(index))
            self.shared_dict[index] = torch.tensor(index)
        return self.shared_dict[index]

    def __len__(self):
        return self.length


# Init
manager = Manager()
shared_dict = manager.dict()
dataset = MyDataset(shared_dict, length=100)

loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=6,
    shuffle=True,
    pin_memory=True
)

# First loop will add data to the shared_dict
for x in loader:
    print(x)

# The second loop will just get the data
for x in loader:
    print(x)
