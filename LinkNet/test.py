'''
-*- coding: utf-8 -*-

Test the LinkNet segmentation model.

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

from linknet import LinkNet
import time

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
        anno_tensor[anno_tensor>0] = 1
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)

        return img_tensor, anno_tensor
    
    def __len__(self):
        return len(self.imgs)
    

test_dl = data.DataLoader(
    PortraitMatteDataset(test_imgs, test_anno),
    batch_size=64,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LinkNet().to(device)

model_path = r'E:\Study\python_code\pytorch\LinkNet\models\best_model.pth'

model.load_state_dict(torch.load(model_path))

num=3

image, mask = next(iter(test_dl))
start_time = time.time()
pred_mask = model(image.to(device))
end_time = time.time()
fmp = 128/(end_time-start_time)

plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(num, 3, i*num+1)
    plt.imshow(image[i].permute(1,2,0).cpu().numpy())
    plt.subplot(num, 3, i*num+2)
    plt.imshow(mask[i].cpu().numpy())
    plt.subplot(num, 3, i*num+3)
    plt.imshow(torch.argmax(pred_mask[i].permute(1,2,0), axis=-1).cpu().detach().numpy())

print("Fame per second for in_size = (256, 256): ", fmp)
plt.show()