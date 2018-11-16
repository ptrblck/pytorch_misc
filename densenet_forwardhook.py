"""
Use forward hooks to get intermediate activations from densenet121.
Create additional conv layers to process these activations to get a
desired number of output channels

@author: ptrblck
"""

import torch
import torch.nn as nn

from torchvision import models


activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

# Create Model
model = models.densenet121(pretrained=False)

# Register forward hooks with name
for name, child in model.features.named_children():
    if 'denseblock' in name:
        print(name)
        child.register_forward_hook(get_activation(name))

# Forward pass
x = torch.randn(1, 3, 224, 224)
output = model(x)

# Create convs to get desired out_channels
out_channels = 1
convs = {'denseblock1': nn.Conv2d(256, out_channels, 1,),
         'denseblock2': nn.Conv2d(512, out_channels, 1),
         'denseblock3': nn.Conv2d(1024, out_channels, 1),
         'denseblock4': nn.Conv2d(1024, out_channels, 1)}

# Apply conv on each activation
for key in activations:
    act = activations[key]
    act = convs[key](act)
    print(key, act.shape)
