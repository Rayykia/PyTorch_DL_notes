'''
Display an example of the dataset.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
import os

import glob
from PIL import Image

plt.figure()
plt.subplot(1,2,1)
pil_img = Image.open(
    r'../hk/training/00001.png'
)
np_img = np.array(pil_img)
plt.imshow(np_img)


plt.subplot(1,2,2)
pil_img_matte = Image.open(
    r'../hk/training/00001_matte.png'
)
np_img_matte = np.array(pil_img_matte)
np_img_matte[np_img_matte<128] = 0
np_img_matte[np_img_matte>=128] = 1
plt.imshow(np_img_matte)
plt.show()
