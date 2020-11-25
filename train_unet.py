# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import json
import cv2
import skimage.draw
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensor
from tqdm import tqdm
from time import time
from PIL import Image
import segmentation_models_pytorch as smp

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


model = smp.Unet('efficientnet-b4', 4, 'imagenet', True, decoder_channels=[128, 64, 32, 16], activation='sigmoid')

class MaskSet(Dataset):
  def __init__(self, dir_img, j_mask, list_files, aug=None, prepro=None):
    self.preprocessing = prepro
    self.dir_img = dir_img
    self.augmentation = aug
    self.j_mask = j_mask
    self.list_keys = j_mask.keys()
    self.dir_files = list_files

  def __len__(self):
      return len(self.dir_files)

  def __getitem__(self, idx):
    name_file = self.dir_files[idx]
    full_path = os.path.join(self.dir_img, name_file)
    img = cv2.imread(full_path)
    h, w, _ = img.shape
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for name in self.list_keys:
      if name_file in name:

        mask = np.zeros((h,w), dtype=np.uint8)

        for region in range(len(self.j_mask[name]['regions'])):
          attr = self.j_mask[name]['regions'][region]['shape_attributes']
          try:
            rr, cc = skimage.draw.polygon(attr['all_points_y'], attr['all_points_x'])
            mask[rr, cc] = 255
          except:
            continue

        break

    things = {'orig_image': full_path, 'size': (h, w)}
    
    if self.augmentation:
          sample = self.augmentation(image=image, mask=mask)
          image, mask = sample['image'], sample['mask']

    return image, mask, things

def augmentations(train=False):
  transforms = []
  if train:
    transforms.extend([
                      albu.ChannelShuffle(p=0.7),
                      albu.CLAHE(p=0.7),
                      albu.Cutout(70, 20,20, p=1.),
                      albu.HueSaturationValue(),
                      albu.HorizontalFlip(),
                      albu.Blur(10, p=0.7),
                      #albu.GaussNoise(p=0.6),
                      #albu.RandomCrop(256,256, p=0.01),
                      albu.Rotate(10, p=1),
                      
      ])
  
  transforms.extend([
                albu.Resize(256, 256),
                ToTensor()
  ])
  return albu.Compose(transforms)

# BIG train dataset (6k)
batch_size = 4

dir_train = 'Nomnet/train'
list_train = os.listdir(dir_train)


with open('Cars6k(new).json', 'r') as j:
  j_obj = json.load(j)

train_set = MaskSet(dir_train, j_obj, list_train, aug=augmentations(True))
train_loader = DataLoader(train_set, batch_size, shuffle=True)

#Test dataset 500
# batch_size = 6

# val_dir = '/content/val'
# val_list = os.listdir(val_dir)
#
# with open('/content/drive/My Drive/val500.json', 'r') as j:
#   j_obj = json.load(j)
#
# val_set = MaskSet(val_dir, j_obj, val_list, aug=augmentations(False))
# val_loader = DataLoader(val_set, batch_size, shuffle=True)

x, y, Imgsize = next(iter(train_loader))

nomer_mask = True

plt.figure(figsize=(16,20))

for i in range(4):
  plt.subplot(3,3,i+1)
  plt.axis('off')
  img = x[i].numpy().transpose(1,2,0)
  if nomer_mask:
    img[y[i].numpy().squeeze() > 0] = 0.2,0.8,0.6
  plt.imshow(img)

model = model.to(device)

optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-3)
criterion = nn.BCELoss()
sheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.3)

#model.load_state_dict(torch.load('/content/drive/My Drive/WeightNet/model(1.0).pth'))

min_iou = 0.9

for epoch in range(15):
  model.train()

  loss_item = []

  for x, y, Imgsize in tqdm(train_loader):
    x, y = x.to(device), y.to(device)

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_item.append(loss.item())
  
  

  coef = 0.6
  count_unit = 0
  summ_iou = 0

  model.eval()
  with torch.no_grad():
    for x, y, Img_size in val_loader:
      count_unit += x.size(0)
      output = model(x.to(device))
      for item in range(x.size(0)):
        dice = torch.sum((output[item] > coef)[y[item] == 1] * 2. / ((torch.sum(output[item] > coef)) + torch.sum(y[item])))
        summ_iou += dice


  total_iou = summ_iou / count_unit
  if total_iou > min_iou:
    min_iou = total_iou
    torch.save(model.state_dict(), '/content/drive/My Drive/WeightNet/model(1.4).pth')
    print(f'Weights saves! best iou: {min_iou}')
  else:
    print(f'iou: {total_iou}')
    

  print(f'epoch: {epoch + 1}, loss: {np.mean(loss_item)}')

model.eval()

with torch.no_grad():
  x, y, Img_size = next(iter(valid_loader))
  output = model(x.to(device))

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(16,50))

for item in range(2):
  ax[item, 0].imshow(x[item].detach().cpu().numpy().transpose(1,2,0))
  ax[item, 1].imshow(output[item].detach().cpu().numpy().transpose(1,2,0).squeeze() > 0.6)

img_numb = 5

a = time()
w, h = Img_size['size'][0][img_numb], Img_size['size'][1][img_numb]
img_mask =cv2.resize((output[img_numb].detach().cpu().numpy().transpose(1,2,0).squeeze() > 0.5).astype(np.uint8), dsize=(h, w))
path_name = Img_size['orig_image'][img_numb]

image_orig = np.array(Image.open(path_name))

contours, hierarchy = cv2.findContours(img_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_orig_copy = image_orig.copy()
    # отображаем контуры поверх изображения
#cv2.drawContours(image_orig_copy, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
#cv2_imshow(image_orig_copy)



model.eval()
x, y, Img_size = next(iter(test_loader))
output = model(x.to(device))

fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(16,25))

for item in range(6):
  ax[item, 0].imshow(x[item].detach().cpu().numpy().transpose(1,2,0))
  ax[item, 1].imshow(y[item].detach().cpu().numpy().transpose(1,2,0).squeeze())
  ax[item, 2].imshow(output[item].detach().cpu().numpy().transpose(1,2,0).squeeze() > 0.6)

#torch.save(model.state_dict(), '/content/drive/My Drive/WeightNet/model(7).pth')

#torch.save(model, '/content/drive/My Drive/WeightNet/model.pt')

# Metrics
model.eval()
coef = 0.6
count_unit = 0
summ_iou = 0

for x, y, Img_size in test_loader:
  count_unit += x.size(0)
  output = model(x.to(device))
  for item in range(x.size(0)):
    dice = torch.sum((output[item] > coef)[y[item] == 1] * 2. / ((torch.sum(output[item] > coef)) + torch.sum(y[item])))
    summ_iou += dice

print(summ_iou / count_unit)
