"""
Permute image data so that channel values of each pixel are flattened to an image patch around the pixel.

@author: ptrblck
"""

import torch

B, C, H, W = 2, 16, 4, 4
# Create dummy input with same values in each channel
x = torch.arange(C)[None, :, None, None].repeat(B, 1, H, W)
print(x)
# Permute channel dimension to last position and view as 4x4 windows
x = x.permute(0, 2, 3, 1).view(B, H, W, 4, 4)
print(x)
# Permute "window dims" with spatial dims, view as desired output
x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, 4*H, 4*W)
print(x)
