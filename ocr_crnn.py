# -*- coding: utf-8 -*-

import cv2
from torch.utils.data import DataLoader, Dataset
import json
import albumentations as albu
from albumentations.pytorch import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from warpctc_pytorch import CTCLoss
import torch.utils.data
import os
import utils
import re
import crnn

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
temp = r'[a-z][0-9]{3}[a-z]{2}[0-9]{2,3}'


class OCR_set(Dataset):

  def __init__(self, root, transform=None):
    super(OCR_set, self).__init__()

    self.root = root
    
    self.img_dirpath = os.path.join(root, 'img')
    self.ann_dirpath = os.path.join(root, 'ann')
    self.img_files = os.listdir(self.img_dirpath)
    self.ann_list = os.listdir(self.ann_dirpath)
    self.ready_img = [i for i in self.img_files if i.split('.')[0] + '.json' in self.ann_list]
    self.transform = transform

  def __len__(self):
    return len(self.ready_img)

  def __getitem__(self, idx):

    img_name = self.ready_img[idx]    
    img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dirpath, img_name)), cv2.COLOR_BGR2RGB)
    json_filepath = os.path.join(self.ann_dirpath, img_name.split('.')[0] + '.json')
    
    if os.path.exists(json_filepath):
      description = json.load(open(json_filepath, 'r'))
      label = description['description'].lower()
    
    
    if self.transform:
      image = self.transform(image=img)['image']

    return image, label

def augmentations(train=False):
  aug = []

  if train:
    aug.extend([
         
         albu.RGBShift(p=0.7),
         albu.Rotate(limit=6, p=1),
         albu.Cutout(30, 2, 2, p=1)

          ])
  aug.extend([
         albu.Resize(32, 128),
         albu.CLAHE(p=1.),
         ToTensor()
  ])

  return albu.Compose(aug)

batch_size = 16
dataset = OCR_set('/content/Plate', augmentations(False))
dataloader = DataLoader(dataset, batch_size, shuffle=True)

x, y = next(iter(dataloader))

plt.figure(figsize=(20, 10))
for i in range(batch_size):

  plt.subplot(4, 4, i+1)
  img_ = x[i].numpy().transpose(1,2,0)
  plt.imshow(img_)


########### confusion


test_set = OCR_set('/content/Plate', augmentations(False))
test_loader = DataLoader(test_set, 1, shuffle=True)
assert test_set

alphabet = '1234567890abekmhopctyx'

nclass = len(alphabet) + 1
nc = 3
nh = 256
imgH = 64
preds_size = torch.tensor([33])

converter = utils.strLabelConverter(alphabet)

crnn = crnn.CRNN(imgH, nc, nclass, nh).to(device)
crnn.load_state_dict(torch.load('/content/drive/My Drive/WeightNet/OCR(3.0)'))

tp_1, fp_1, fn_1 = 0, 0, 0 # True Positive, False positive, False negative for first head
tp_2, fp_2, fn_2 = 0, 0, 0 # for double head
result = ['', '']

with torch.no_grad():
  for x, y in test_loader:
      
    cpu_images, cpu_texts = x.to(device), y
    preds = crnn(cpu_images)
    preds = preds.view(-1, preds_size.item(), 1, 23)

    for idx, head in enumerate(preds):

      _, preds1 = head.max(2)
      preds1 = preds1.transpose(1, 0).contiguous().view(-1)
      raw_pred = converter.decode(preds1.data, preds_size.data, raw=False)

      postpro = re.findall(temp, raw_pred)
      sim_pred = postpro[0] if postpro != [] else 'Unknown'

      result[idx] = sim_pred
    
    if result[0] == 'Unknown':
      fp_1 += 1
      fp_2 += 1
    elif result[0] == y[0]:
      tp_1 += 1
      if result[0] == result[1]:
        tp_2 += 1
      else:
        fp_2 += 1
    elif result[0] != y[0]:
      fn_1 += 1
      if result[0] == result[1]:
        fn_2 += 1


torch.save(crnn.state_dict(), '/content/drive/My Drive/WeightNet/OCR(1.6)')
