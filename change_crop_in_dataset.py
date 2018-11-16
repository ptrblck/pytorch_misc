"""
Change the crop size on the fly using a Dataset.
MyDataset.set_state(stage) switches between crop sizes.
Alternatively, the crop size could be specified.

@author: ptrblck
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self):
        self.images = [TF.to_pil_image(x) for x in torch.ByteTensor(10, 3, 48, 48)]
        self.set_stage(0)

    def __getitem__(self, index):
        image = self.images[index]

        # Switch your behavior depending on stage
        image = self.crop(image)
        x = TF.to_tensor(image)
        return x

    def set_stage(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
            self.crop = transforms.RandomCrop((32, 32))
        elif stage == 1:
            print('Using (28, 28) crops')
            self.crop = transforms.RandomCrop((28, 28))

    def __len__(self):
        return len(self.images)


dataset = MyDataset()
loader = DataLoader(dataset,
                    batch_size=2,
                    num_workers=2,
                    shuffle=True)

# Use standard crop size
for batch_idx, data in enumerate(loader):
    print('Batch idx {}, data shape {}'.format(
        batch_idx, data.shape))

# Switch to stage1 crop size
loader.dataset.set_stage(1)

# Check the shape again
for batch_idx, data in enumerate(loader):
    print('Batch idx {}, data shape {}'.format(
        batch_idx, data.shape))
