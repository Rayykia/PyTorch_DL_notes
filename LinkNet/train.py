'''
-*- coding: utf-8 -*-

Train the LinkNet segmentation model.

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
        anno_tensor[anno_tensor>0] = 1
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)

        return img_tensor, anno_tensor
    
    def __len__(self):
        return len(self.imgs)
    
train_dl = data.DataLoader(
    PortraitMatteDataset(imgs, anno),
    batch_size=128,
    shuffle=True
)

test_dl = data.DataLoader(
    PortraitMatteDataset(test_imgs, test_anno),
    batch_size=128,
)


# #display the batch
# img_batch, anno_batch = next(iter(train_dl))
# plt.figure(figsize=(24,8))
# for i in range(4):
#     img = img_batch[i].permute(1, 2, 0).numpy()
#     anno = anno_batch[i].numpy()
#     plt.subplot(2, 4, i+1)
#     plt.imshow(img)
#     plt.subplot(2, 4, i+5)
#     plt.imshow(anno)
# plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LinkNet().to(device)

# # Test the model.
# img_batch, _ = next(iter(train_dl))
# print(model(img_batch.to(device)).shape)

model_train = SemSegFit(
    epoches=40,
    train_dataloader=train_dl,
    test_dataloader=test_dl,
    model=model
)

model_train.train()
model_train.plot()

'''
epoch: 0, train_loss:0.61766, train_acc:66.16%, train_IoU:0.559, test_loss:0.70861, test_acc:41.91% test_IoU:0.219.
epoch: 1, train_loss:0.35651, train_acc:86.90%, train_IoU:0.815, test_loss:0.53920, test_acc:74.38% test_IoU:0.687.
epoch: 2, train_loss:0.27358, train_acc:90.40%, train_IoU:0.862, test_loss:0.35854, test_acc:86.24% test_IoU:0.809.
epoch: 3, train_loss:0.22055, train_acc:92.19%, train_IoU:0.886, test_loss:0.28885, test_acc:89.08% test_IoU:0.843.
epoch: 4, train_loss:0.19113, train_acc:93.17%, train_IoU:0.900, test_loss:0.24673, test_acc:90.56% test_IoU:0.858.
epoch: 5, train_loss:0.17272, train_acc:93.70%, train_IoU:0.907, test_loss:0.27735, test_acc:88.61% test_IoU:0.832.
epoch: 6, train_loss:0.15556, train_acc:94.35%, train_IoU:0.917, test_loss:0.22609, test_acc:90.79% test_IoU:0.864.
epoch: 7, train_loss:0.14144, train_acc:94.90%, train_IoU:0.924, test_loss:0.17357, test_acc:93.63% test_IoU:0.902.
epoch: 8, train_loss:0.13148, train_acc:95.37%, train_IoU:0.931, test_loss:0.16646, test_acc:94.01% test_IoU:0.908.
epoch: 9, train_loss:0.12731, train_acc:95.53%, train_IoU:0.933, test_loss:0.16247, test_acc:94.18% test_IoU:0.910.
epoch:10, train_loss:0.12295, train_acc:95.74%, train_IoU:0.936, test_loss:0.15992, test_acc:94.28% test_IoU:0.912.
epoch:11, train_loss:0.11967, train_acc:95.86%, train_IoU:0.938, test_loss:0.15945, test_acc:94.30% test_IoU:0.912.
epoch:12, train_loss:0.11692, train_acc:95.97%, train_IoU:0.939, test_loss:0.15514, test_acc:94.42% test_IoU:0.914.
epoch:13, train_loss:0.11357, train_acc:96.13%, train_IoU:0.942, test_loss:0.15467, test_acc:94.46% test_IoU:0.915.
epoch:14, train_loss:0.11096, train_acc:96.23%, train_IoU:0.944, test_loss:0.15398, test_acc:94.50% test_IoU:0.916.
epoch:15, train_loss:0.11096, train_acc:96.23%, train_IoU:0.943, test_loss:0.15353, test_acc:94.51% test_IoU:0.916.
epoch:16, train_loss:0.11013, train_acc:96.27%, train_IoU:0.944, test_loss:0.15325, test_acc:94.52% test_IoU:0.916.
epoch:17, train_loss:0.11008, train_acc:96.28%, train_IoU:0.944, test_loss:0.15339, test_acc:94.52% test_IoU:0.916.
epoch:18, train_loss:0.10909, train_acc:96.32%, train_IoU:0.945, test_loss:0.15321, test_acc:94.53% test_IoU:0.916.
epoch:19, train_loss:0.10871, train_acc:96.33%, train_IoU:0.945, test_loss:0.15290, test_acc:94.54% test_IoU:0.917.
epoch:20, train_loss:0.10942, train_acc:96.29%, train_IoU:0.944, test_loss:0.15281, test_acc:94.55% test_IoU:0.917.
epoch:21, train_loss:0.10824, train_acc:96.35%, train_IoU:0.945, test_loss:0.15273, test_acc:94.55% test_IoU:0.917.
epoch:22, train_loss:0.10840, train_acc:96.34%, train_IoU:0.945, test_loss:0.15228, test_acc:94.57% test_IoU:0.917.
epoch:23, train_loss:0.10894, train_acc:96.33%, train_IoU:0.945, test_loss:0.15230, test_acc:94.57% test_IoU:0.917.
epoch:24, train_loss:0.10839, train_acc:96.34%, train_IoU:0.945, test_loss:0.15240, test_acc:94.56% test_IoU:0.917.
epoch:25, train_loss:0.10826, train_acc:96.35%, train_IoU:0.945, test_loss:0.15251, test_acc:94.56% test_IoU:0.917.
epoch:26, train_loss:0.10859, train_acc:96.33%, train_IoU:0.945, test_loss:0.15232, test_acc:94.56% test_IoU:0.917.
epoch:27, train_loss:0.10883, train_acc:96.32%, train_IoU:0.945, test_loss:0.15229, test_acc:94.56% test_IoU:0.917.
epoch:28, train_loss:0.10868, train_acc:96.33%, train_IoU:0.945, test_loss:0.15237, test_acc:94.56% test_IoU:0.917.
epoch:29, train_loss:0.10786, train_acc:96.37%, train_IoU:0.946, test_loss:0.15227, test_acc:94.56% test_IoU:0.917.
epoch:30, train_loss:0.10853, train_acc:96.33%, train_IoU:0.945, test_loss:0.15234, test_acc:94.56% test_IoU:0.917.
epoch:31, train_loss:0.10795, train_acc:96.37%, train_IoU:0.945, test_loss:0.15257, test_acc:94.56% test_IoU:0.917.
epoch:32, train_loss:0.10842, train_acc:96.35%, train_IoU:0.945, test_loss:0.15249, test_acc:94.56% test_IoU:0.917.
epoch:33, train_loss:0.10852, train_acc:96.33%, train_IoU:0.945, test_loss:0.15241, test_acc:94.56% test_IoU:0.917.
epoch:34, train_loss:0.10808, train_acc:96.36%, train_IoU:0.945, test_loss:0.15239, test_acc:94.56% test_IoU:0.917.
epoch:35, train_loss:0.10842, train_acc:96.34%, train_IoU:0.945, test_loss:0.15238, test_acc:94.56% test_IoU:0.917.
epoch:36, train_loss:0.10849, train_acc:96.34%, train_IoU:0.945, test_loss:0.15245, test_acc:94.56% test_IoU:0.917.
epoch:37, train_loss:0.10782, train_acc:96.37%, train_IoU:0.946, test_loss:0.15242, test_acc:94.56% test_IoU:0.917.
epoch:38, train_loss:0.10895, train_acc:96.31%, train_IoU:0.945, test_loss:0.15244, test_acc:94.56% test_IoU:0.917.
epoch:39, train_loss:0.10828, train_acc:96.35%, train_IoU:0.945, test_loss:0.15238, test_acc:94.56% test_IoU:0.917.
Done
'''