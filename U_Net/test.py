'''
-*- coding: utf-8 -*-

U-Net segmentation on test dataset.

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

model_path = r'E:\Study\python_code\pytorch\U_Net\models\best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetModel().to(device)
model.load_state_dict(torch.load(model_path))


all_test_pics = glob.glob(
    r'E:/Study/python_code/pytorch/datasets/hk/testing/*.png'
)
test_imgs = [p for p in all_test_pics if 'matte' not in p]
test_anno = [p for p in all_test_pics if 'matte' in p]

# dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class PortraitMatteDataset(data.Dataset):
    def __init__(self, img_path, anno_path):
        super().__init__()
        self.imgs = img_path
        self.annos = anno_path

    def __getitem__(self, index):
        img = self.imgs[index]
        anno = self.annos[index]

        pil_img = Image.open(img)
        pil_img = pil_img.convert('RGB')
        img_tensor = transform(pil_img)

        anno_img = Image.open(anno)
        anno_tensor = transform(anno_img)
        anno_tensor[anno_tensor<0.5] = 0
        anno_tensor[anno_tensor>=0.5] = 1
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)

        return img_tensor, anno_tensor
    
    def __len__(self):
        return len(self.imgs)
    

test_dl = data.DataLoader(
    PortraitMatteDataset(test_imgs, test_anno),
    batch_size=8,
)


img_batch, matte_batch = next(iter(test_dl))

# 24 pics = 4 * 6
matte_pred = model(img_batch.to(device))


plt.figure()
for i, (img, matte) in enumerate(
    zip(
        img_batch, matte_batch
    )
):
    plt.subplot(4, 6, i*3+1)
    plt.imshow(img.permute(1,2,0).numpy())
    plt.subplot(4, 6, i*3+2)
    plt.imshow(matte.numpy())
    plt.subplot(4, 6, i*3+3)
    plt.imshow(torch.argmax(matte_pred[i].permute(1,2,0), -1).detach().cpu().numpy())
plt.show()