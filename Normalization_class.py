#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:03:42 2022

@author: acina@iit.local
"""

import numpy as np
import stain_utils as utils
from scipy.signal import wiener
from skimage import filters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from segmentation_utils import *

def calculate_stain_matrix(img, beta=0.15, alpha=1, method = 'transparency'):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    h, w, _ = img.shape

    '''GET STAIN MATRIX'''
    #convert from RGB to OD and reshape to have a matrix of shape (w*h)*3 channels
    od = utils.RGB_to_OD(img).reshape((-1, 3))   
    
    if method == 'gabor':
        thr = 0.9
        kernel_size = (5, 5)  #depend on how big are the features to capture
        sigma = 1
        thetas = np.arange(0, 2 * np.pi, np.pi / 4)
        
        #thetas = np.arange(0, 360, 45)
        lambd = 10 * np.pi / 180
        gamma = 1.2 #aspect ratio (circular or elliptical kernel). If I go towards 0 --> ellipsis
        ps = 0      #phase of the wave
        
        mask = utils.gabor_filter(img, kernel_size, sigma, thetas, lambd, gamma, ps, thr)
        
        od_hat = od[mask.reshape(-1) == False]
    elif method == 'otsu':
        mask = utils.otsu_filter(img)
        od_hat = od[mask.reshape(-1) == False]
    elif method == 'transparency':
        beta = beta
        od_hat = od[(od > beta).any(axis=1), :]
    
    eigvals, eigvecs = np.linalg.eig(np.cov(od_hat, rowvar=False))
    #verify eigvals and vecs
    a = eigvecs.dot(np.diag(eigvals).dot(np.linalg.inv(eigvecs)))
    np.testing.assert_almost_equal(a, np.cov(od_hat, rowvar=False))

    #find the 2 eigen values with highest values and extract the 2 most important eigvectors
    ind = np.argsort(np.abs(eigvals))[::-1]
    eigvecs = eigvecs[:, ind][:, :2]#extract only the most important eigenvectors

    if eigvecs[0, 0] < 0: eigvecs[:, 0] *= -1
    if eigvecs[0, 1] < 0: eigvecs[:, 1] *= -1

    #project the optical density in the new space
    That = np.dot(od_hat, eigvecs)
    #normalize rows to project the points on a unit sphere
    That = utils.normalize_rows(That)

    phi = np.arctan2(That[:, 1], That[:, 0])
    
    minPhi = np.percentile(phi, alpha)

    maxPhi = np.percentile(phi, 100 - alpha)
    
    v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]).T)
    v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]).T)
    
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    
    return HE
    


class Normalizer(object):
    """
    class to normalize images. The fit method is used to calculate the stain matrix of the reference. 
    This matrix will be used for all the other images as a reference
    """

    def __init__(self, beta=0.15, alpha=1, method = 'transparency', scan = False, standardize_bright = True):
        self.beta = beta #threshold for transparency. Used only if method is 'transparency'
        self.alpha = alpha #percentile to calculate the extreme values in eigenspace 1 means 1 and 99 percentiles
        self.method = method #otsu, gabor or transparency to segment object in the image
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.scan = scan #use SCAN algorithm to refine stain matrix calculation
        self.standardize_bright = standardize_bright

    def fit(self, target):
        if self.standardize_bright:
            target = utils.standardize_brightness(target)
        #calculate stain matrix
        self.stain_matrix_target = calculate_stain_matrix(target, self.beta, self.alpha, self.method)
        #calculate stain concentrations
        self.target_concentrations = utils.calculate_concentrations(target, self.stain_matrix_target)
        #refine the stain matrix calculation if scan = True
        if self.scan:
            
            self.stain_matrix_target, self.target_concentrations = self.refine_stain_matrix(target,
                                                                                            self.stain_matrix_target,
                                                                                            self.target_concentrations)
        
    def refine_stain_matrix(self, target, stain, concentrations):
        
        if self.standardize_bright:
            target = utils.standardize_brightness(target)
        
        h, w, c = target.shape
        
        #calculate hematoxylin image in grayscale color space
        H = concentrations[:, 0].reshape(h, w)
        h_gray = 10**(-1 * H)   
        #calculate eosin image in grayscale color space
        E = concentrations[:, 1].reshape(h, w)
        e_gray = 10**(-1 * E)
        
        #normalize hematoxylin to have full range 0-255
        hem_norm = utils.min_max_norm(h_gray, 0, 255).astype(int)        
        #wiener filter
        filtered_img = wiener(hem_norm, (5, 5)).astype(int)
        #threshold image with Otsu to find nuclei
        threshold = filters.threshold_otsu(filtered_img)            
        #create a mask to extract only foreground objects
        nuclei_mask = filtered_img < threshold
        
        #stroma extraction
        d_gray = h_gray - e_gray
        #negative values = 0
        d_gray[d_gray < 0] = 0
        #calculate maximum and minimum of intensities to use them as cluster centers for initialization
        maximum = np.max(d_gray)
        minimum = np.min(d_gray)        
        initialization = np.array([[maximum, minimum]]).reshape(-1, 1)
        #Kmeans
        kmeans = KMeans(n_clusters = 2, init = initialization).fit(d_gray.reshape(-1, 1))
        kmeans.labels_
        #calculate mean intensities of clusters
        cl1 = d_gray.reshape(-1, 1)[kmeans.labels_ == 0] #mean intensity is 73
        cl2 = d_gray.reshape(-1, 1)[kmeans.labels_ == 1] #mean intensity is 210
        
        mean_intensity_cl1 = np.mean(cl1)
        mean_intensity_cl2 = np.mean(cl2)
        #stroma has highest intensity
        if mean_intensity_cl1 > mean_intensity_cl2:
            max_intensity = 0
        else:
            max_intensity = 1
        
        stroma_mask = np.zeros(d_gray.reshape(-1, 1).shape)
        stroma_mask[kmeans.labels_ == max_intensity] = 1
        stroma_mask = stroma_mask.reshape(d_gray.shape).astype(int)
        stroma_mask = stroma_mask == 1
        
        #FINAL STAIN SEPARATION
        #compute median RGB values of nuclei and stroma using segmentation masks
        nuclei_median = np.median(target[nuclei_mask].reshape((-1, 3)), axis = 0)
        stroma_median = np.median(target[stroma_mask].reshape((-1, 3)), axis = 0)
        
        #convert to OD to compare with previous values of stain
        od_nuclei = utils.RGB_to_OD(nuclei_median)
        od_stroma = utils.RGB_to_OD(stroma_median)
        
        #concatenate
        stain = np.stack((od_nuclei, od_stroma), axis = 0)
        #normalize
        stain = utils.normalize_rows(stain)
        #calculate new concentrations
        concentrations = utils.calculate_concentrations(target, stain)
        
        return stain, concentrations
        
    def transform(self, I):
        if self.standardize_bright:
            I = utils.standardize_brightness(I)
        #transform new image using target image stain matrix as reference
        #stain matrix of source image
        stain_matrix_source = calculate_stain_matrix(I)
        #concentration of source image
        source_concentrations = utils.calculate_concentrations(I, stain_matrix_source)
        #if scan refine stain and concentration calculation
        if self.scan:
            
            stain_matrix_source, source_concentrations = self.refine_stain_matrix(I,
                                                                                  stain_matrix_source,
                                                                                  source_concentrations)
        #find 99 percentile of source image concentration
        maxC_source = np.percentile(source_concentrations, 99, axis = 0).reshape((1, 2))
        #find 99 percentile of target image concentration
        maxC_target = np.percentile(self.target_concentrations, 99, axis = 0).reshape((1, 2))
        #rescale source concentration
        source_concentrations *= (maxC_target / maxC_source)
        
        #normalize image
        Inorm = np.multiply(255, 10**(-self.stain_matrix_target.T.dot(source_concentrations.T)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, newshape = I.shape).astype(np.uint8)
        return Inorm

    def hematoxylin_gray(self, I):
        if self.standardize_bright:
            I = utils.standardize_brightness(I)
        
        h, w, c = I.shape
        stain_matrix_source = calculate_stain_matrix(I)
        source_concentrations = utils.calculate_concentrations(I, stain_matrix_source)
        
        if self.scan:
            stain_matrix_source, source_concentrations = self.refine_stain_matrix(I,
                                                                                  stain_matrix_source,
                                                                                  source_concentrations)
        
        H = source_concentrations[:, 0].reshape(h, w)
        H = 10**(-1 * H)
        return np.multiply(255, H).astype(np.uint8)
    
    def hematoxylin_color(self, I):
        
        if self.standardize_bright:
            I = utils.standardize_brightness(I)
        
        h, w, c = I.shape
        norm_he = calculate_stain_matrix(I)
        C = utils.calculate_concentrations(I, norm_he)
        if self.scan:
            norm_he, C = self.refine_stain_matrix(I,
                                                  norm_he,
                                                  C)
        #plot hematoxylin in RGB space (not gray scale)
        h_color = np.dot(C[:, 0][:, None], norm_he.T[:, 0][None, :])
        h_color = 10**(-1 * h_color)
        H_color = np.multiply(255, h_color)
        H_color[H_color > 255] = 254
        H_color = np.reshape(H_color, (h, w, 3)).astype(np.uint8)
        
        return H_color
    
    def eosin_gray(self, I):
        
        if self.standardize_bright:
            I = utils.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = calculate_stain_matrix(I)
        source_concentrations = utils.calculate_concentrations(I, stain_matrix_source)
        if self.scan:
            stain_matrix_source, source_concentrations = self.refine_stain_matrix(I,
                                                                                  stain_matrix_source,
                                                                                  source_concentrations)
        E = source_concentrations[:, 1].reshape(h, w)
        E = 10**(-1 * E)
        return np.multiply(255, E).astype(np.uint8)
    
    def eosin_color(self, I):
        
        if self.standardize_bright:
            I = utils.standardize_brightness(I)
        h, w, c = I.shape
        norm_he = calculate_stain_matrix(I)
        C = utils.calculate_concentrations(I, norm_he)
        if self.scan:            
            norm_he, C = self.refine_stain_matrix(I,
                                                  norm_he,
                                                  C)
        #plot hematoxylin in RGB space (not gray scale)
        e_color = np.dot(C[:, 1][:, None], norm_he.T[:, 1][None, :])
        e_color = 10**(-1 * e_color)
        E_color = np.multiply(255, e_color)
        E_color[E_color > 255] = 254
        E_color = np.reshape(E_color, (h, w, 3)).astype(np.uint8)
        
        return E_color
    
    def segment_MANA(self, img, threshold_max = 150, thr_small = 0.25, thr_big = 1.5):
        if self.standardize_bright:
            img = utils.standardize_brightness(img)
            
        h_gray = self.hematoxylin_gray(img)
        
        i = preprocess(h_gray)

        pwm_curve, bins_count = pwm(i)

        pwm_fitted = fit_pwm(pwm_curve, bins_count)

        thrs = calculate_thresholds(pwm_fitted, threshold_max = threshold_max)

        median_areas = compute_areas(i, thrs)

        initial_mask, initial_threshold = compute_initial_mask(i, thrs, median_areas)

        small_areas, normal_areas, big_areas, props = compute_min_max_normal_regions(initial_mask,
                                                                                     min_perc = thr_small,
                                                                                     max_perc = thr_big)

        images_ws = segment_big_areas(i, initial_threshold, big_areas, props)

        normal_a = find_normal_areas(i, props, normal_areas)

        ws = segment_normal_remove_small(images_ws, normal_a)
        
        return ws