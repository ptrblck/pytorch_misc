"""
Permute all pixels of MNIST data and try to learn it using simple model.

@author: ptrblck
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms

import numpy as np

# Create random indices to permute images
indices = np.arange(28*28)
np.random.shuffle(indices)


def shuffle_image(tensor):
    tensor = tensor.view(-1)[indices].view(1, 28, 28)
    return tensor


# Apply permuatation using transforms.Lambda
train_dataset = datasets.MNIST(root='./data',
                               download=False,
                               train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   transforms.Lambda(shuffle_image)
                                ]))

test_dataset = datasets.MNIST(root='./data',
                              download=False,
                              train=False,
                              transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   transforms.Lambda(shuffle_image)
                               ]))

train_loader = DataLoader(train_dataset,
                          batch_size=1,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 4, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7*7*8, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


def train():
    acc = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, dim=1)
        accuracy = (pred == target).sum() / float(pred.size(0))
        acc += accuracy.data.float()

        if (batch_idx + 1) % 10 == 0:
            print('batch idx {}, loss {}'.format(
                batch_idx, loss.item()))

    acc /= len(train_loader)
    print('Train accuracy {}'.format(acc))


def test():
    acc = 0.0
    losses = 0.0
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output, dim=1)

            accuracy = (pred == target).sum() / float(pred.size(0))
            acc += accuracy.data.float()
            losses += loss.item()
    acc /= len(test_loader)
    losses /= len(test_loader)
    print('Acc {}, loss {}'.format(
        acc, losses))


model = MyModel()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train()
test()

# Visualize filters
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
filts1 = model.conv1.weight.data
grid = make_grid(filts1)
grid = grid.permute(1, 2, 0)
plt.imshow(grid)
