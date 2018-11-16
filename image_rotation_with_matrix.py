"""
Rotate image given an angle.
1. Calculate rotated position for each input pixel
2. Use meshgrid and rotation matrix to achieve the same

@author: ptrblck
"""

import torch
import numpy as np


# Create dummy image
im = torch.zeros(1, 1, 10, 10)
im[:, :, :, 2] = 1.

# Set angle
angle = torch.tensor([72 * np.pi / 180.])

# Calculate rotation for each target pixel
x_mid = (im.size(2) + 1) / 2.
y_mid = (im.size(3) + 1) / 2.
im_rot = torch.zeros_like(im)
for r in range(im.size(2)):
    for c in range(im.size(3)):
        x = (r - x_mid) * torch.cos(angle) + (c - y_mid) * torch.sin(angle)
        y = -1.0 * (r - x_mid) * torch.sin(angle) + (c - y_mid) * torch.cos(angle)
        x = torch.round(x) + x_mid
        y = torch.round(y) + y_mid

        if (x >= 0 and y >= 0 and x < im.size(2) and y < im.size(3)):
            im_rot[:, :, r, c] = im[:, :, x.long(), y.long()]


# Calculate rotation with inverse rotation matrix
rot_matrix = torch.tensor([[torch.cos(angle), torch.sin(angle)],
                          [-1.0*torch.sin(angle), torch.cos(angle)]])

# Use meshgrid for pixel coords
xv, yv = torch.meshgrid(torch.arange(im.size(2)), torch.arange(im.size(3)))
xv = xv.contiguous()
yv = yv.contiguous()
src_ind = torch.cat((
    (xv.float() - x_mid).view(-1, 1),
    (yv.float() - y_mid).view(-1, 1)),
    dim=1
)

# Calculate indices using rotation matrix
src_ind = torch.matmul(src_ind, rot_matrix.t())
src_ind = torch.round(src_ind)
src_ind += torch.tensor([[x_mid, y_mid]])

# Set out of bounds indices to limits
src_ind[src_ind < 0] = 0.
src_ind[:, 0][src_ind[:, 0] >= im.size(2)] = float(im.size(2)) - 1
src_ind[:, 1][src_ind[:, 1] >= im.size(3)] = float(im.size(3)) - 1

# Create new rotated image
im_rot2 = torch.zeros_like(im)
src_ind = src_ind.long()
im_rot2[:, :, xv.view(-1), yv.view(-1)] = im[:, :, src_ind[:, 0], src_ind[:, 1]]
im_rot2 = im_rot2.view(1, 1, 10, 10)

print('Using method 1: {}'.format(im_rot))
print('Using method 2: {}'.format(im_rot2))
