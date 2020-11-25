# -*- coding: utf-8 -*-

import segmentation_models_pytorch as smp
import numpy as np
import torch
import utils
from PIL import Image
import albumentations as albu
from albumentations.pytorch import ToTensor
import cv2
from time import time
import re
import os
from skimage.morphology import convex_hull_image
import rectDetector
import telebot
import random

temp = r'[a-z][0-9]{3}[a-z]{2}[0-9]{2,3}'

rectDetect = rectDetector.RectDetector()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model = smp.Unet('efficientnet-b4', activation='sigmoid', encoder_weights='imagenet', encoder_depth=4, decoder_channels=[128, 64, 32, 16]).to(device)

print('loading model unet...\n')
model.load_state_dict(torch.load('model(1.0).pth', map_location='cpu'))

def prepare_image(path_name):
  image_orig = cv2.imread(path_name)
  img_shape = image_orig.shape
  image = augmentations()(image=image_orig)['image']
  
  return image.unsqueeze(0), image_orig, img_shape


def augmentations():
    transforms = []
    transforms.extend([
                albu.Resize(256,256),
                ToTensor()
  ])
    return albu.Compose(transforms)

nomer_aug = albu.Compose([
                      albu.Resize(32, 128),
                      albu.CLAHE(always_apply=True),
                      albu.Normalize(),
                      albu.RandomBrightness((0.2, 0.4), always_apply=True),
                      ToTensor()
])



full_path_image = '/content/drive/My Drive/DataCars'
#list_cars = os.listdir(full_path_image)
list_cars = []

import crnn2 as crnn


alphabet = '1234567890abekmhopctyx'

nclass = len(alphabet) + 1
nc = 3
nh = 256
imgH = 64
preds_size = torch.tensor([33])
converter = utils.strLabelConverter(alphabet)
crnn = crnn.CRNN(imgH, nc, nclass, nh).to(device)
print('loading crnn model...\n')
crnn.load_state_dict(torch.load('OCR(3.0)', map_location='cpu'))
crnn.eval()

def detect(path_name_img):

  image, image_orig, img_shape = prepare_image(path_name_img)
  test_output = model(image.to(device))

  w, h, _ = img_shape
  mask_gray = cv2.resize((test_output[0].cpu().detach().numpy().squeeze() > 0.6).astype(np.uint8), dsize=(h, w))
  mask_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
  contours, hierarchy = cv2.findContours(mask_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  count_contours = len(contours)
  outputs = []
  mask_list = []
  nomer_list = []

  for cont in range(count_contours):
  # делаем несколько масок
    mask = np.zeros_like(mask_gray)
    mask = cv2.drawContours(mask, [contours[cont]], -1, (255,0,0), 3, cv2.LINE_AA, np.expand_dims(hierarchy[:, cont], axis=1), 1)
    mask = convex_hull_image(mask).astype(np.uint8)
    if np.sum(mask) < 1500:
      continue

    mask_list.append(mask)
    a3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    cord = rectDetect.detect([a3 * 255])
# x1, x2, x3, x4 = cord[0]

    zones = rectDetect.get_cv_zonesBGR(image_orig.copy(), cord)
    outputs.append(zones[0])



  for nomer in outputs:
    image = nomer_aug(image=nomer)['image'].unsqueeze(0)
    preds1 = crnn(image.to(device))
    preds1 = preds1.view(-1, 33, 1, 23)
    conf = [0, 0]

    for idx, head in enumerate(preds1):
      _, preds = head.max(2)
      batch = preds.size(1)
      preds = preds.transpose(1, 0).contiguous().view(-1)

      sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
      postpro = re.findall(temp, sim_pred)
    
      sim_pred = postpro[0] if postpro != [] else 'Unknown'
      conf[idx] = sim_pred

    # if conf[0] == conf[1]:
    #   nomer_list.append(conf[0])
    # else:
    #   nomer_list.append('Unknown')
    nomer_list.append(conf)

  return nomer_list, outputs, mask_rgb, image_orig 

def Detecting(chat_id, path, flag=False):
  try:
    a = time()
    text_list, zones_nomer, mask_, image_arr = detect(path)
    print(time() - a, 'time detect')
  except:
    return bot.send_message(chat_id, 'Не найден номер')
  b = time()
  image_ = Image.fromarray(image_arr[..., ::-1])
  if flag:
    bot.send_photo(chat_id, image_, caption='Исходная картинка')
  for k_nomer in range(len(zones_nomer)):
    if text_list[k_nomer][0] != text_list[k_nomer][1]:
      caption_ = f'First model: {text_list[k_nomer][0].upper()}\nSecond model: {text_list[k_nomer][1].upper()}'
    else:
      caption_ = text_list[k_nomer][0].upper()
    bot.send_photo(chat_id, Image.fromarray(zones_nomer[k_nomer]), caption=caption_)
  mask_ = Image.fromarray(mask_ * 255)
  bot.send_photo(chat_id, mask_, reply_markup=markup)
  print(time() - b, 'time send photo')

print('Telegram bot activate!\n')

bot = telebot.TeleBot('1200244813:AAF_DPJcNp4G4TO4fbBdTOEpq6n-_IFROV0')

@bot.message_handler(commands=['start'])
def start_message(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, 'Привет, чтобы распознать номер - отправь изображение')


@bot.message_handler(content_types=['photo'])
def detect_plate(message):
    chat_id = message.chat.id
    file = bot.get_file(message.photo[-1].file_id)

    downloaded_file = bot.download_file(file.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    path_name = 'image.jpg'
    a = time()
    Detecting(chat_id, path_name, flag=False)
    print(time() - a)


@bot.message_handler(content_types=['text'])
def send_text(message):
    chat_id = message.chat.id
    if message.text.lower() == 'привет':
        bot.send_message(chat_id, 'Привет, жду сообщений')
    else:
      path_name = os.path.join(full_path_image, list_cars[random.choice(range(len(list_cars)))])
      Detecting(chat_id, path_name, flag=True)



bot.polling()