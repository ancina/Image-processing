#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:21:38 2022

@author: acina@iit.local
"""

import stain_utils as utils
from Normalization_class import Normalizer
import matplotlib.pyplot as plt
from segmentation_utils import *

path_abs = '/home/acina/Documents/image analysis/JC_2022_04_20/SCAN algorithm dataset/'
path_img = 'IMAGES/IMAGES/BREAST/Breast H&E (20x)/'

name_src = '142I17_13.tif'

#name_src = '/home/acina/Documents/image analysis/images_lab/normal intestinal epithelium CRC patient 3.png'
img = utils.read_image(path_abs + path_img + name_src)
#img = utils.read_image(name_src)


beta = 0.15
alpha = 1
method = 'transparency'

norm = Normalizer(beta, alpha, method, scan = True)

ws = norm.segment_MANA(img)

plot_img(img, ws)

norm.fit(img)

img_norm = norm.transform(img)

hem = norm.hematoxylin_color(img)
eos = norm.eosin_color(img)

plt.figure(figsize=(20, 12))
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_norm)
plt.show()


plt.figure(figsize=(20, 12))
plt.subplot(1,2,1)
plt.imshow(eos)
plt.subplot(1,2,2)
plt.imshow(hem)
plt.show()


target = '/home/acina/Documents/image analysis/images_lab/tumoral lymph node patient 7.png'

img_s = utils.read_image(target)

img_n = norm.transform(img_s)

plt.figure(figsize=(20, 12))
plt.subplot(1,2,1)
plt.imshow(img_s)
plt.subplot(1,2,2)
plt.imshow(img_n)
plt.show()