#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 08:29:52 2022

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

def preprocess(img_hematoxylin, kernel = (3, 3), sigma = 1):
    
    im = utils.min_max_norm(img_hematoxylin, min_ = 0, max_ = 255)

    im = cv.GaussianBlur(im, kernel, sigmaX = sigma)
    
    return im

def pwm(img):
    # getting data of the histogram
    count, bins_count = np.histogram(img.ravel(), density = False, bins = 256)
    #   
    # # finding the PDF of the histogram using count values
    pixels = np.arange(256)

    pwm_list = []
    for i in pixels:
        pwm_list.append((np.sum(count[:i] * pixels[:i])) / np.sum(count[:i]))
        
    pwm = np.array(pwm_list)
    pwm[0] = 0.0 #nan replaced by 0
    
    return pwm, bins_count

def fit_pwm(pwm, bins_count, degree = 15):
    coefficients = np.polyfit(bins_count[1:], pwm, full = False, deg = degree)

    fitted = np.polyval(coefficients, bins_count[1:])
    
    return fitted

def calculate_thresholds(pwm_fitted, threshold_max):
    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(pwm_fitted))

    # find switching points
    thresholds = np.where(np.diff(np.sign(smooth_d2)))[0]

    ind_thresh = np.where(thresholds < threshold_max)[0]

    thresholds = thresholds[ind_thresh]
    
    return thresholds

def compute_areas(img, thresholds):
    kernel = np.ones((3, 3),np.uint8)
    medians = []
    for thr in thresholds:
        mask = (img < thr).astype(np.uint8)

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
    
    return medians

def compute_initial_mask(im, thresholds, medians):
    kernel = np.ones((3, 3),np.uint8)
    
    initial_threshold = thresholds[np.argmax(medians)]

    initial_mask = (im < initial_threshold).astype(np.uint8)

    area_t = initial_mask.shape[0] * initial_mask.shape[1] *0.1

    initial_mask = cv.dilate(initial_mask, kernel, iterations = 1)

    initial_mask = morphology.area_closing(initial_mask, area_t)
    
    return initial_mask, initial_threshold

def compute_min_max_normal_regions(initial_mask, min_perc = 0.25, max_perc = 1.5):
    img_label = label(initial_mask)

    props = regionprops(img_label)

    areas = np.array([proper['area'] for proper in props])

    mean_area = np.mean(areas)

    minimum = min_perc * mean_area
    maximum = max_perc * mean_area
    
    small_areas = np.where(areas < minimum)[0]
    big_areas = np.where(areas > maximum)[0]
    n1 = areas > minimum
    n2 = areas < maximum

    normal_areas = np.where(n1 * n2 == True)[0]
    
    return small_areas, normal_areas, big_areas, props

def segment_big_areas(img, initial_threshold, big_areas, props):
    kernel = np.ones((3, 3),np.uint8)
    images_ws = np.zeros(img.shape, dtype = np.uint8)
    ind_ws = 0  #I have to increase the starting point of watershed at each iteration
    thr = initial_threshold
    for ind in big_areas:
        sub_prop = np.array(props)[ind]
        
        minr, minc, maxr, maxc = sub_prop['bbox']
            
        crop = img[minr:maxr, minc:maxc]
        
        mask_ = (crop < thr).astype(np.uint8)
        
        mask_ = cv.dilate(mask_, kernel, iterations = 1)
        
        mask_ = morphology.area_closing(mask_, 40)
        
        distance = distance_transform_edt(mask_)
        
        markers = label(peak_local_max(distance, min_distance = 3, indices=False), connectivity = 2)
        
        ws = watershed(-distance, markers, mask = mask_)
        
        plt.imshow(ws, 'jet')
        plt.show()
        
        ind_not_zero = np.nonzero(ws)
        
        images_ws[minr:maxr, minc:maxc][ind_not_zero] = ind_ws + ws[ind_not_zero]
        
        ind_ws += np.max(ws)
        
    return images_ws

def find_normal_areas(im, props, normal_areas):
    
    normal_a = np.zeros(im.shape, dtype = np.uint8)
    for i in normal_areas:
        sub_prop = np.array(props)[i]
            
        indices = sub_prop['coords']       
            
        normal_a[indices[:, 0], indices[:, 1]] = 1
        
    return normal_a

def segment_normal_remove_small(images_ws, normal_a):
    normal_a = label(normal_a)

    normal_a = normal_a + np.max(images_ws)

    normal_a[normal_a == 255] = 0

    images_ws = images_ws + normal_a

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
            
    return remove_small_ws

def plot_img(img, ws):
    plt.figure(figsize=(20, 12))
    plt.imshow(mark_boundaries(img, ws))
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    