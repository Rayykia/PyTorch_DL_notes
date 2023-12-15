'''
-*- coding: utf-8 -*-

Train the U-Net segmentation model.

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
from modelfit import SemSegFit

all_pics = glob.glob(
    r'../hk/training/*.png'
)

imgs = [p for p in all_pics if 'matte' not in p]
anno = [p for p in all_pics if 'matte' in p]
# len(imgs) = len(anno) = 1662

# train test split
np.random.seed(2021)
index = np.random.permutation(len(imgs))
imgs = np.array(imgs)[index]
anno = np.array(anno)[index]

all_test_pics = glob.glob(
    r'../hk/testing/*.png'
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
    
train_dl = data.DataLoader(
    PortraitMatteDataset(imgs, anno),
    batch_size=8,
    shuffle=True
)

test_dl = data.DataLoader(
    PortraitMatteDataset(test_imgs, test_anno),
    batch_size=8,
)


# #display the batch
img_batch, anno_batch = next(iter(train_dl))
plt.figure(figsize=(24,8))
for i in range(4):
    img = img_batch[i].permute(1, 2, 0).numpy()
    anno = anno_batch[i].numpy()
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.subplot(2, 4, i+5)
    plt.imshow(anno)
plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetModel().to(device)

# # Test the model.
# img_batch, _ = next(iter(train_dl))
# print(model(img_batch.to(device)).shape)

model_train = SemSegFit(
    epoches=30,
    train_dataloader=train_dl,
    test_dataloader=test_dl,
    model=model
)

model_train.train()
model_train.plot()

# epoch: 0, train_loss:0.58880, train_acc:69.20%, test_loss:0.42224, test_acc:82.53%
# epoch: 1, train_loss:0.37043, train_acc:84.18%, test_loss:0.34140, test_acc:85.55%
# epoch: 2, train_loss:0.34561, train_acc:85.25%, test_loss:0.34187, test_acc:85.37%
# epoch: 3, train_loss:0.33799, train_acc:85.63%, test_loss:0.31745, test_acc:86.57%
# epoch: 4, train_loss:0.33341, train_acc:85.83%, test_loss:0.34952, test_acc:85.09%
# epoch: 5, train_loss:0.32459, train_acc:86.30%, test_loss:0.31977, test_acc:86.50%
# epoch: 6, train_loss:0.31817, train_acc:86.61%, test_loss:0.32463, test_acc:86.27%
# epoch: 7, train_loss:0.29622, train_acc:87.57%, test_loss:0.29155, test_acc:87.71%
# epoch: 8, train_loss:0.28232, train_acc:88.27%, test_loss:0.28058, test_acc:88.28%
# epoch: 9, train_loss:0.27273, train_acc:88.75%, test_loss:0.27337, test_acc:88.67%
# epoch:10, train_loss:0.26366, train_acc:89.16%, test_loss:0.27363, test_acc:88.57%
# epoch:11, train_loss:0.25715, train_acc:89.42%, test_loss:0.25715, test_acc:89.32%
# epoch:12, train_loss:0.24866, train_acc:89.83%, test_loss:0.25303, test_acc:89.47%
# epoch:13, train_loss:0.24289, train_acc:90.06%, test_loss:0.26455, test_acc:89.09%
# epoch:14, train_loss:0.23215, train_acc:90.54%, test_loss:0.24370, test_acc:90.00%
# epoch:15, train_loss:0.22847, train_acc:90.72%, test_loss:0.24178, test_acc:90.09%
# epoch:16, train_loss:0.22686, train_acc:90.79%, test_loss:0.24068, test_acc:90.14%
# epoch:17, train_loss:0.22538, train_acc:90.85%, test_loss:0.23967, test_acc:90.18%
# epoch:18, train_loss:0.22440, train_acc:90.90%, test_loss:0.23837, test_acc:90.26%
# epoch:19, train_loss:0.22344, train_acc:90.94%, test_loss:0.23761, test_acc:90.27%
# epoch:20, train_loss:0.22234, train_acc:90.99%, test_loss:0.23599, test_acc:90.34%
# epoch:21, train_loss:0.22054, train_acc:91.05%, test_loss:0.23593, test_acc:90.35%
# epoch:22, train_loss:0.22033, train_acc:91.06%, test_loss:0.23604, test_acc:90.36%
# epoch:23, train_loss:0.22013, train_acc:91.07%, test_loss:0.23593, test_acc:90.36%
# epoch:24, train_loss:0.22008, train_acc:91.08%, test_loss:0.23594, test_acc:90.37%
# epoch:25, train_loss:0.21993, train_acc:91.09%, test_loss:0.23585, test_acc:90.37%
# epoch:26, train_loss:0.21974, train_acc:91.09%, test_loss:0.23579, test_acc:90.38%
# epoch:27, train_loss:0.21968, train_acc:91.10%, test_loss:0.23569, test_acc:90.38%
# epoch:28, train_loss:0.21943, train_acc:91.10%, test_loss:0.23569, test_acc:90.38%
# epoch:29, train_loss:0.21955, train_acc:91.10%, test_loss:0.23568, test_acc:90.38%
# Done