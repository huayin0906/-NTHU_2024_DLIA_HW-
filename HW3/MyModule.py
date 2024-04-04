#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, x, y, transform = None):
        self.x = x
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        new_x = np.transpose(self.x[idx], (1, 2, 0))
        return self.transform(Image.fromarray(new_x)), self.y[idx]

