'''
-*- coding: utf-8 -*-

U-Net segmentation on random dataset.

Dataset: Deep Automatic Portrait Matting (The Chinese University of Hong Kong)

Author: Rayykia
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

from unet_module import UNetModel

all_pics = glob.glob(
    r'.\internet_data\*.jpg'
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img_batch = torch.tensor([])

for i in all_pics:
    pil_img = Image.open(i)
    img_tensor = torch.unsqueeze(transform(pil_img), 0)
    img_batch = torch.cat([img_batch, img_tensor], 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetModel().to(device)

model_path = r'.\models\best_model.pth'

model.load_state_dict(torch.load(model_path))

matte_pred = model(img_batch.to(device))

plt.figure()
for i, image in enumerate(img_batch):
    plt.subplot(2, 4, i*2+1)
    plt.imshow(image.permute(1,2,0).numpy())
    plt.subplot(2, 4, i*2+2)
    plt.imshow(torch.argmax(matte_pred[i].permute(1,2,0), -1).cpu().numpy())
plt.show()