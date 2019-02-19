"""
Shows how to store and load data from redis using a PyTorch
Dataset and DataLoader (with multiple workers).

@author: ptrblck
"""

import redis

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np


# Create random data and push to redis
r = redis.Redis(host='localhost', port=6379, db=0)

nb_images = 100
for idx in range(nb_images):
    # Use long for the fake images, as it's easier to store the target with it
    data = np.random.randint(0, 256, (3, 24, 24), dtype=np.long).tobytes()
    target = bytes(np.random.randint(0, 10, (1,)).astype(np.long))
    r.set(idx, data + target)


# Create RedisDataset
class RedisDataset(Dataset):
    def __init__(self,
                 redis_host='localhost',
                 redis_port=6379,
                 redis_db=0,
                 length=0,
                 transform=None):

        self.db = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.length = length
        self.transform = transform

    def __getitem__(self, index):
        data = self.db.get(index)
        data = np.frombuffer(data, dtype=np.long)
        x = data[:-1].reshape(3, 24, 24).astype(np.uint8)
        y = torch.tensor(data[-1]).long()
        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.length


# Load samples from redis using multiprocessing
dataset = RedisDataset(length=100, transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=2,
    shuffle=True
)

for data, target in loader:
    print(data.shape)
    print(target.shape)
