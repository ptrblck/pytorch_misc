"""
Combine Conv3d with an RNN Module.
Use windowed frames as inputs.

@author: ptrblck
"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MyModel(nn.Module):
    def __init__(self, window=16):
        super(MyModel, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.MaxPool3d((1, 2, 2)),
            nn.ReLU()
        )

        self.rnn = nn.RNN(
            input_size=6*16*12*12,
            hidden_size=1,
            num_layers=1,
            batch_first=True
        )
        self.hidden = torch.zeros(1, 1, 1)
        self.window = window

    def forward(self, x):
        self.hidden = torch.zeros(1, 1, 1)  # reset hidden

        activations = []
        for idx in range(0, x.size(2), self.window):
            x_ = x[:, :, idx:idx+self.window]
            x_ = self.conv_model(x_)
            x_ = x_.view(x_.size(0), 1, -1)
            activations.append(x_)
        x = torch.cat(activations, 1)
        out, hidden = self.rnn(x, self.hidden)

        return out, hidden


class MyDataset(Dataset):
    '''
    Returns windowed frames from sequential data.
    '''
    def __init__(self, frames=512):
        self.data = torch.randn(3, 2048, 24, 24)
        self.frames = frames

    def __getitem__(self, index):
        index = index * self.frames
        x = self.data[:, index:index+self.frames]
        return x

    def __len__(self):
        return self.data.size(1) / self.frames


model = MyModel()
dataset = MyDataset()
x = dataset[0]
output, hidden = model(x.unsqueeze(0))
