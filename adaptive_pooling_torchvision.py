"""
Adaptive pooling layer examples

@author: ptrblck
"""

import torch
import torch.nn as nn

import torchvision.models as models


# Use standard model with [batch_size, 3, 224, 224] input
model = models.vgg16(pretrained=False)
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224)
output = model(x)

# Try bigger input
x_big = torch.randn(batch_size, 3, 299, 299)
try:
    output = model(x_big)
except RuntimeError as e:
    print(e)

# Try smaller input
x_small = torch.randn(batch_size, 3, 128, 128)
try:
    output = model(x_small)
except RuntimeError as e:
    print(e)
# Both don't work, since we get a size mismatch for these sizes

# Get the size of the last activation map before the classifier
def size_hook(module, input, output):
    print(output.shape)

model.features[-1].register_forward_hook(size_hook)
output = model(x)

# We see that the last pooling layer returns an activation of
# [batch_size, 512, 7, 7]. So let's replace it with an adaptive layer with an
# output shape of 7x7.
model.features[-1] = nn.AdaptiveMaxPool2d(output_size=7)

# Now let's try the other shapes again
output = model(x_big)
output = model(x_small)

x_tiny = torch.randn(batch_size, 3, 16, 16)
output = model(x_tiny)

# Now these inputs are working!
# There is however a minimal size as we need a spatial size of at least 1x1
# to pass into the adaptive pooling layer
x_too_small = torch.randn(batch_size, 3, 15, 15)
try:
    output = model(x_too_small)
except RuntimeError as e:
    print(e)
