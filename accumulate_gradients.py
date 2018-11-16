"""
Comparison of accumulated gradients/losses to vanilla batch update.
Comments from @albanD:
https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20

@author: ptrblck
"""

import torch
import torch.nn as nn


# Accumulate loss for each samples
# more runtime, more memory
x1 = torch.ones(2, 1)
w1 = torch.ones(1, 1, requires_grad=True)
y1 = torch.ones(2, 1) * 2

criterion = nn.MSELoss()

loss1 = 0
for i in range(10):
    output1 = torch.matmul(x1, w1)
    loss1 += criterion(output1, y1)
loss1 /= 10  # scale loss to match batch gradient
loss1.backward()

print('Accumulated losses: {}'.format(w1.grad))

# Use whole batch to calculate gradient
# least runtime, more memory
x2 = torch.ones(20, 1)
w2 = torch.ones(1, 1, requires_grad=True)
y2 = torch.ones(20, 1) * 2

output2 = torch.matmul(x2, w2)
loss2 = criterion(output2, y2)
loss2.backward()
print('Batch gradient: {}'.format(w2.grad))

# Accumulate scaled gradient
# more runtime, least memory
x3 = torch.ones(2, 1)
w3 = torch.ones(1, 1, requires_grad=True)
y3 = torch.ones(2, 1) * 2

for i in range(10):
    output3 = torch.matmul(x3, w3)
    loss3 = criterion(output3, y3)
    loss3 /= 10
    loss3.backward()

print('Accumulated gradient: {}'.format(w3.grad))
