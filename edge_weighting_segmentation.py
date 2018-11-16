"""
Apply weighting to edges for a segmentation task

@author: ptrblck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# Create dummy input and target with two squares
output = F.log_softmax(torch.randn(1, 3, 24, 24), 1)
target = torch.zeros(1, 24, 24, dtype=torch.long)
target[0, 4:12, 4:12] = 1
target[0, 14:20, 14:20] = 2
plt.imshow(target[0])

# Edge calculation
# Get binary target
bin_target = torch.where(target > 0, torch.tensor(1), torch.tensor(0))
plt.imshow(bin_target[0])

# Use average pooling to get edge
o = F.avg_pool2d(bin_target.float(), kernel_size=3, padding=1, stride=1)
plt.imshow(o[0])

edge_idx = (o.ge(0.01) * o.le(0.99)).float()
plt.imshow(edge_idx[0])

# Create weight mask
weights = torch.ones_like(edge_idx, dtype=torch.float32)
weights_sum0 = weights.sum()  # Save initial sum for later rescaling
weights = weights + edge_idx * 2.  # Weight edged with 2x loss
weights_sum1 = weights.sum()
weights = weights / weights_sum1 * weights_sum0  # Rescale weigths
plt.imshow(weights[0])

# Calculate loss
criterion = nn.NLLLoss(reduction='none')
loss = criterion(output, target)
loss = loss * weights  # Apply weighting
loss = loss.sum() / weights.sum()  # Scale loss
