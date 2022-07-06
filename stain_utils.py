#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:54:44 2022

@author: acina@iit.local
"""

import numpy as np
import cv2 as cv
import spams
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import morphology, filters

##########################################

def calculate_concentrations(img, norm_he):
    od = RGB_to_OD(img).reshape((-1, 3))
    #norm_he.T shape is 3x2
    C = spams.lasso(od.T, D = norm_he.T, mode = 2, lambda1 = 0.01, pos = True).toarray()
    
    return C.T

def min_max_norm(img, min_, max_):
    minimum = np.min(img)
    maximum = np.max(img)
    
    img_n = (img - minimum) * ((max_ - min_) / (maximum - minimum)) + min_
    
    return img_n


def otsu_filter(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    threshold = filters.threshold_otsu(gray)
        
    #create a mask to extract only foreground objects
    mask = gray > threshold
    
    return mask

def gabor_filter(img, kernel_size, sigma, thetas, lambd, gamma, ps, thr):
    
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    gabor_kernels = []
    gabor_imgs = []
    
    for i, theta in enumerate(thetas):
        gabor = cv.getGaborKernel(kernel_size, sigma, theta, lambd, gamma, ps, ktype = cv.CV_32F) #store 32 bit float
        img_gabor = cv.filter2D(gray, ddepth = cv.CV_8UC3, kernel = gabor)
        
        gabor_kernels.append(gabor)
        
        gabor_imgs.append(img_gabor)
        
    img_final = np.array(gabor_imgs)

    img_norm = img_final.sum(0)
    
    img_norm = min_max_norm(img_norm, 0, 255)

    mask = img_norm > (np.max(img_norm) * thr)

    return mask

def read_image(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)

    contours,hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv.boundingRect(cnt)

    im = im[y:y+h,x:x+w]
    return im


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return (-1 * np.log10((I)/ 255)).astype(np.float)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * 10**(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]
