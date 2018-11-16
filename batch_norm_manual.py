"""
Comparison of PyTorch BatchNorm layers and a manual calculation

@author: ptrblck
"""

import torch
import torch.nn as nn

# Init BatchNorm2d
bn2 = nn.BatchNorm2d(3)
bn2.weight.data.fill_(0.1)
bn2.bias.data.zero_()

# Create 2D tensor
tmp = torch.cat((torch.ones(1, 2, 1), torch.ones(1, 2, 1) * 2), 2)
x2 = torch.cat((torch.zeros(1, 2, 2), tmp, tmp * 2), 0)
x2.unsqueeze_(0)

# Calculate stats
x2_mean = x2.mean(-1).mean(-1)
num_elem2 = 4
x2_var_unbiased = ((x2 - x2_mean.view(1, 3, 1, 1))**2).sum(2).sum(2) / (num_elem2 - 1)
print('x2: ', x2)
print('x2 mean: ', x2_mean)
print('x2 var_unbiased: ', x2_var_unbiased)
print('bn2 running_mean: ', bn2.running_mean)
print('bn2 running_var: ', bn2.running_var)
print('Expected bn2 running_mean after forward pass: ',
      bn2.running_mean * (1 - bn2.momentum) + x2_mean * bn2.momentum)
print('Expected bn2 running_var after forward pass: ',
      bn2.running_var * (1 - bn2.momentum) + x2_var_unbiased * bn2.momentum)

# Perform forward pass on 2D data
output2 = bn2(x2)
print('bn2 running mean after forward pass:', bn2.running_mean)
print('bn2 running var after forward pass:', bn2.running_var)

# Init BatchNorm3d
bn3 = nn.BatchNorm3d(3)
bn3.weight.data.fill_(0.1)
bn3.bias.data.zero_()

# Create 3D tensor from 2D
x3 = x2.unsqueeze(2).repeat(1, 1, 5, 1, 1)

# Calculate stats
x3_mean = x3.mean(-1).mean(-1).mean(-1)
num_elem3 = 5 * 4
x3_var_unbiased = ((x3 - x3_mean.view(1, 3, 1, 1, 1))**2).sum(2).sum(2).sum(2) / (num_elem3 - 1)
print('x3: ', x3)
print('x3 mean: ', x3_mean)
print('x3 var_unbiased: ', x3_var_unbiased)
print('bn3 running_mean: ', bn3.running_mean)
print('bn3 running_var: ', bn3.running_var)
print('Expected bn3 running_mean after forward pass: ',
      bn3.running_mean * (1 - bn3.momentum) + x3_mean * bn3.momentum)
print('Expected bn3 running_var after forward pass: ',
      bn3.running_var * (1 - bn3.momentum) + x3_var_unbiased * bn3.momentum)

# Perform forward pass on 3D data
output3 = bn3(x3)
print('bn3 running mean after forward pass:', bn3.running_mean)
print('bn3 running var after forward pass:', bn3.running_var)
