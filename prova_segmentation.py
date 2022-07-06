#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:19:24 2022

@author: acina@iit.local
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import stain_utils as utils
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.measure import label
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from Normalization_class import Normalizer

#path = '/mnt/c/Users/acina/Documents/cmp3VdA/JC/JC_2022_04_20/experiment/data/'
#path_abs = '/home/acina/Documents/image analysis/JC_2022_04_20/SCAN algorithm dataset/'
#path = 'IMAGES/IMAGES/BREAST/Breast H&E (20x)/'
path_abs = '/home/acina/Documents/image analysis/'
path_img = 'images_lab/'
name_src = 'tumoral intestinal epithelium CRC patient 3.png'

#name_src = '/home/acina/Documents/image analysis/images_lab/tumoral lymph node patient 7.png'
img = utils.read_image(path_abs + path_img + name_src)

r, g, b = cv.split(img)

plt.imshow(g, cmap = 'gray')
plt.title('Channel')
plt.show()

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

h, s, v = cv.split(img_hsv)

plt.imshow(v, cmap = 'gray')
plt.title('Channel')
plt.show()



beta = 0.15
alpha = 1
method = 'transparency'

norm = Normalizer(beta, alpha, method, scan = True)

h_gray = norm.hematoxylin_gray(img)

im = utils.min_max_norm(b, min_ = 0, max_ = 255)

im = cv.GaussianBlur(im, (3, 3), sigmaX = 1)

plt.imshow(im, cmap = 'gray')
plt.title('Channel')
plt.show()

# getting data of the histogram
count, bins_count = np.histogram(im.ravel(), density = False, bins = 256)
#   
# # finding the PDF of the histogram using count values
pdf = count / sum(count)
#   
# # using numpy np.cumsum to calculate the CDF
# # We can also find using the PDF values by looping and adding
cdf = np.cumsum(pdf)
#   
# # plotting PDF and CDF
plt.figure()
plt.plot(bins_count[1:], pdf, color="red", label="PDF")
plt.plot(bins_count[1:], cdf, label="CDF")
plt.legend()


pixels = np.arange(256)

pwm_list = []
for i in pixels:
    pwm_list.append((np.sum(count[:i] * pixels[:i])) / np.sum(count[:i]))
    
plt.figure()
plt.plot(bins_count[1:], pwm_list, color="red", label="PWM")
plt.grid()
plt.legend()

pwm = np.array(pwm_list)
pwm[0] = 0.0

coefficients = np.polyfit(bins_count[1:], pwm, full = False, deg = 15)

fitted = np.polyval(coefficients, bins_count[1:])

plt.figure()
plt.plot(bins_count[1:], pwm, color="blue", label="PWM")
plt.plot(bins_count[1:], fitted, color="r", label="PWM fitted", linestyle='dashed')
plt.grid()
plt.legend()

# compute second derivative
smooth_d2 = np.gradient(np.gradient(fitted))

# find switching points
thresholds = np.where(np.diff(np.sign(smooth_d2)))[0]

ind_thresh = np.where(thresholds < 150)[0]

thresholds = thresholds[ind_thresh]

plt.figure()
plt.plot(bins_count[1:], fitted, color="r", label="PWM fitted", linestyle='dashed')
plt.grid()
plt.legend()
for i, infl in enumerate(thresholds, 1):
    plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
plt.legend(bbox_to_anchor=(1.55, 1.0))
plt.show()

kernel = np.ones((3, 3),np.uint8)
medians = []
for thr in thresholds:
    mask = (im < thr).astype(np.uint8)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask , 2 , cv.CV_32S)

    median_area = np.median(stats[1:, -1])
    
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    mask = morphology.area_closing(mask, 40)
    
    plt.imshow(mask, cmap = 'gray')
    plt.show()
    
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask , 2 , cv.CV_32S)

    median_area = np.median(stats[1:, -1])
    
    medians.append(median_area)


medians = np.array(medians)

plt.plot(thresholds, medians)
plt.scatter(thresholds, medians, c = 'r', label = 'thresholds')
plt.grid()
plt.legend()
plt.show()

initial_threshold = thresholds[np.argmax(medians)]

initial_mask = (im < initial_threshold).astype(np.uint8)

area_t = initial_mask.shape[0] * initial_mask.shape[1] *0.1

initial_mask = cv.dilate(initial_mask, kernel, iterations = 1)

#initial_mask = morphology.area_closing(initial_mask, area_t)


plt.figure(figsize=(20, 12))
plt.imshow(initial_mask, cmap = 'gray')
plt.show()

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, initial_mask))
plt.show()

img_label = label(initial_mask)

props = regionprops(img_label)

areas = np.array([proper['area'] for proper in props])

mean_area = np.mean(areas)

plt.hist(areas, bins = 'auto')
plt.vlines(x = mean_area, ymin = 0, ymax = 200, color = 'r')
plt.show()

minimum = 0.25 * mean_area
maximum = 1.5 * mean_area


small_areas = np.where(areas < minimum)[0]

remove_small = initial_mask.copy()
for i in small_areas:
        
        sub_prop = np.array(props)[i]
        
        indices = sub_prop['coords']       
        
        remove_small[indices[:, 0], indices[:, 1]] = 0

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, remove_small))
plt.show()

big_areas = np.where(areas > maximum)[0]

big_a = np.zeros(im.shape, dtype = np.uint8)
centroids = []
for i in big_areas:
        
        sub_prop = np.array(props)[i]
        
        indices = sub_prop['coords']       
        
        centroids.append(sub_prop['centroid']  )
        
        big_a[indices[:, 0], indices[:, 1]] = 1

centroids = np.array(centroids)

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, big_a))
#plt.scatter(centroids[:, 0], centroids[:, 1], color = 'r')
plt.title('big areas')
plt.show()

small_a = np.zeros(im.shape, dtype = np.uint8)
for i in small_areas:
        
        sub_prop = np.array(props)[i]
        
        indices = sub_prop['coords']       
        
        small_a[indices[:, 0], indices[:, 1]] = 1

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, small_a))
plt.title('small areas')
plt.show()


n1 = areas > minimum
n2 = areas < maximum

normal_areas = np.where(n1 * n2 == True)[0]

normal_a = np.zeros(im.shape, dtype = np.uint8)
for i in normal_areas:
        
        sub_prop = np.array(props)[i]
        
        indices = sub_prop['coords']       
        
        normal_a[indices[:, 0], indices[:, 1]] = 1

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, normal_a))
plt.title('normal areas')
plt.show()



sigma_spot_detection = 3
sigma_outline = 1


# =============================================================================
# import pyclesperanto_prototype as cle
# 
# images_ws = np.zeros(v.shape, dtype = np.uint8)
# ind_ws = 0  #I have to increase the starting point of watershed at each iteration
# thr = initial_threshold
# for ind in big_areas:
#     sub_prop = np.array(props)[ind]
#     
#     minr, minc, maxr, maxc = sub_prop['bbox']
#         
#     crop = v[minr:maxr, minc:maxc]
#     
#     crop_org = img_src[minr:maxr, minc:maxc]
#     
#     img_gaussian = cle.gaussian_blur(1 - crop, sigma_x=3, sigma_y=3, sigma_z=3)
#     
#     img_maxima_locations = cle.detect_maxima_box(img_gaussian, radius_x=0, radius_y=0, radius_z=0)
#     
#     img_gaussian2 = cle.gaussian_blur(1 - crop, sigma_x=1, sigma_y=1, sigma_z=1)
#     img_thresh = cle.threshold_otsu(img_gaussian2)
#     
#     img_thresh = morphology.area_closing(np.array(img_thresh, dtype = 'uint8'), 20)
#     
#     img_relevant_maxima = cle.binary_and(img_thresh, img_maxima_locations)
#     
#     voronoi_separation = cle.masked_voronoi_labeling(img_relevant_maxima, img_thresh)
#     
#     
#     ws = np.array(voronoi_separation, dtype = 'uint8')
#     
#     #ws = cle.voronoi_otsu_labeling(1 - crop, spot_sigma = sigma_spot_detection, outline_sigma = sigma_outline)
#     
#     plt.imshow(ws, 'jet')
#     plt.show()
#     
#     ind_not_zero = np.nonzero(ws)
#     
#     images_ws[minr:maxr, minc:maxc][ind_not_zero] = ind_ws + ws[ind_not_zero]
#     
#     ind_ws += np.max(ws)
# 
# 
# plt.figure(figsize=(20, 12))
# plt.imshow(mark_boundaries(img_src, images_ws))
# plt.show()
# =============================================================================






'''#######################################################WATERSHED ON CROPS#############################################'''
images_ws = np.zeros(im.shape, dtype = np.uint8)
ind_ws = 0  #I have to increase the starting point of watershed at each iteration
thr = initial_threshold
for ind in big_areas:
    sub_prop = np.array(props)[ind]
    
    minr, minc, maxr, maxc = sub_prop['bbox']
        
    crop = im[minr:maxr, minc:maxc]
    
    crop_org = img[minr:maxr, minc:maxc]
    
    mask_ = (crop < thr).astype(np.uint8)
    
    mask_ = cv.dilate(mask_, kernel, iterations = 1)
    
    mask_ = morphology.area_closing(mask_, 40)
    
    plt.imshow(mask_, cmap = 'gray')
    plt.show()
    #plt.imshow(mask, cmap = 'gray')
    #plt.show()
    
    distance = distance_transform_edt(mask_)
    
    #markers = peak_local_max(distance, min_distance = 10, indices=True)
    
    markers = label(peak_local_max(distance, min_distance = 3, indices=False), connectivity = 2)
    
    ws = watershed(-distance, markers, mask = mask_)
    
    plt.imshow(ws, 'jet')
    plt.show()
    
    ind_not_zero = np.nonzero(ws)
    
    images_ws[minr:maxr, minc:maxc][ind_not_zero] = ind_ws + ws[ind_not_zero]
    
    ind_ws += np.max(ws)


plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, images_ws))
plt.show()

normal_a = label(normal_a)

normal_a = normal_a + np.max(images_ws)

normal_a[normal_a == 255] = 0

images_ws = images_ws + normal_a

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, images_ws))
plt.show()


props = regionprops(images_ws)

areas = np.array([proper['area'] for proper in props])

mean_area_ws = np.mean(areas)

minimum_ws = 0.25 * mean_area_ws

small_areas = np.where(areas < minimum_ws)[0]

remove_small_ws = images_ws.copy()
for i in small_areas:
        
        sub_prop = np.array(props)[i]
        
        indices = sub_prop['coords']       
        
        remove_small_ws[indices[:, 0], indices[:, 1]] = 0

plt.figure(figsize=(20, 12))
plt.imshow(mark_boundaries(img, remove_small_ws))
plt.show()
        



