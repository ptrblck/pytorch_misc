"""
Test implementation of locally connected 2d layer
The first part of the script was used for debugging

@author: ptrblck
"""


import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


## DEBUG
batch_size = 5
in_channels = 3
h, w = 24, 24
x = torch.ones(batch_size, in_channels, h, w)
kh, kw = 3, 3  # kernel_size
dh, dw = 1, 1  # stride
x_windows = x.unfold(2, kh, dh).unfold(3, kw, dw)
x_windows = x_windows.contiguous().view(*x_windows.size()[:-2], -1)

out_channels = 2
weights = torch.randn(1, out_channels, in_channels, *x_windows.size()[2:])
output = (x_windows.unsqueeze(1) * weights).sum([2, -1])
## DEBUG


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


# Create input
batch_size = 5
in_channels = 3
h, w = 24, 24
x = torch.randn(batch_size, in_channels, h, w)

# Create layer and test if backpropagation works
out_channels = 2
output_size = 22
kernel_size = 3
stride = 1
conv = LocallyConnected2d(
    in_channels, out_channels, output_size, kernel_size, stride, bias=True)

out = conv(x)
out.mean().backward()
print(conv.weight.grad)
