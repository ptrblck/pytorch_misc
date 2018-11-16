"""
Usage of WeightedRandomSampler using an imbalanced dataset with
class imbalance 99 to 1.

@author: ptrblck
"""

import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader


# Create dummy data with class imbalance 99 to 1
numDataPoints = 1000
data_dim = 5
bs = 100
data = torch.randn(numDataPoints, data_dim)
target = torch.cat((torch.zeros(int(numDataPoints * 0.99), dtype=torch.long),
                    torch.ones(int(numDataPoints * 0.01), dtype=torch.long)))

print('target train 0/1: {}/{}'.format(
    (target == 0).sum(), (target == 1).sum()))

# Compute samples weight (each sample should get its own weight)
class_sample_count = torch.tensor(
    [(target == t).sum() for t in torch.unique(target, sorted=True)])
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in target])

# Create sampler, dataset, loader
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataset = torch.utils.data.TensorDataset(data, target)
train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

# Iterate DataLoader and check class balance for each batch
for i, (x, y) in enumerate(train_loader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
