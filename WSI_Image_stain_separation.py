#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:29:58 2022

@author: acina@iit.local
"""

import openslide
#from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters
from openslide.deepzoom import DeepZoomGenerator
import os
from pathlib import Path
import glob
import tifffile as tiff
from Normalization_class import Normalizer
import stain_utils as utils
import pyclesperanto_prototype as cle
import cv2 as cv
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
import segmentation_utils as seg_ut
from csbdeep.utils import normalize
from stardist.models import StarDist2D

class WSI_images:
    def __init__(self, path):
        
        svs = openslide.open_slide(path)
        
        self.svs = svs
        self.num_levels = svs.level_count
        self.level_dimensions = svs.level_dimensions
        self.level_downsamples = svs.level_downsamples
        
        print('SVS file infos:')
        print('Levels: ', svs.level_count)
        print('Level dimensions: ', svs.level_dimensions)
        print('Level downsamples: ', svs.level_downsamples)    
        #print('Level properties: ', svs.properties)
        
    def compute_mask(self, level):
        '''
        Compute the mask for a specified level of the wsi.
        the mask will be used to filter out the blank images
        
        level: which level use to compute the mask. 0 is for full resolution
        
        '''
        
        self.level = level
        
        region = self.svs.read_region(location = (0, 0), level = level, size = self.level_dimensions[level])

        region = region.convert('HSV')
        #normalize with max value for a 8 bit image. 
        region_array = np.array(region)

        region_array = region_array[:, :, 1]
        
        print('Computing mask...')

        #otsu calculate a threshold All pixels with an intensity higher than this 
        #value are assumed to be foreground.
        threshold = filters.threshold_otsu(region_array)       

        #create a mask to extract only foreground objects
        mask = region_array > threshold

        #10% of the pixel
        area_threshold = self.level_dimensions[level][0] * self.level_dimensions[level][1] / 100 * 10
        #Area closing removes all dark structures of an image with a surface smaller than area_threshold
        mask = morphology.area_closing(mask, area_threshold)
        #create a mask for the central object in the image
        mask = morphology.remove_small_objects(mask, min_size = 64)
        
        self.mask = mask
        
        print('Mask computed!!')
        
        return mask
    
    def create_save_tiles(self, path_to_save, tile_size = 1024, overlap = 0, plot = False):
        '''
        function to create the tiles from a svs file. tiles is a complete new object that is not linked to svs
        
        path_to_save: folder that will be created to save the tiles
        tile_size: size of each tile. 
        overlap: number of overlapping pixel of contigous tiles
        plot: wheter to plot or not the tiles and the corresponding masks
        
        '''
        
        print('Extracting tiles...')
        #generate the tiles
        tiles = DeepZoomGenerator(self.svs, tile_size = tile_size, overlap = overlap, limit_bounds = False)

        #find the tile number that corresponds to the selected slide dimension
        ind = np.where(np.array((tiles.level_dimensions) - np.array(self.level_dimensions[self.level]) == 1))[0][0]

        tile_number = ind
        #indices that correspond to the grid created for the tiles
        cols, rows = tiles.level_tiles[tile_number]
        
        #up to one folder
        file_path = Path(__file__).parent.absolute()
        #create folders
        img_path = file_path.joinpath(path_to_save)
        #folder for the original tiles
        orig_tiles_path = img_path.joinpath('original')
        Path(str(img_path)).mkdir(parents = True, exist_ok = True) 
        Path(str(orig_tiles_path)).mkdir(parents = True, exist_ok = True) 
        
        #iterate over the grid of tiles
        for row in range(rows):
            for col in range(cols):
                #tile name for saving
                tile_name = orig_tiles_path.joinpath(str(col) + '_' + str(row) + '.tif')
                
                #extract the coordinates of the tile
                (x, y), down, _ = tiles.get_tile_coordinates(tile_number, (col, row))
                tile_w, tile_h = tiles.get_tile_dimensions(tile_number, address = (col, row))                
                #get the tile
                temp_tile = tiles.get_tile(tile_number, (col, row))
                temp_tile_RGB = temp_tile.convert('RGB')
                
                if plot:
                    plt.imshow(temp_tile_RGB)
                    plt.show()    
                
                #downsampling factor of the level considered to generate the mask
                level_scale = int(self.level_downsamples[self.level])
                
                #evaluate the mask (first the row so y coordinate and then column so x coordinate and width)
                t_mask = self.mask[int(np.floor(y * 1 / level_scale)):int(np.floor(y * 1 / level_scale))+int(np.floor(tile_h)),
                                       int(np.floor(x * 1 / level_scale)):int(np.floor(x * 1 / level_scale)) + int(np.floor(tile_w))]
                if plot:
                    plt.imshow(t_mask)
                    plt.show()
                
                pixel_on = np.sum(t_mask) != 0
                
                #if there is a portion of the mask
                if pixel_on != 0:
                    print("Now saving tile with title: ", tile_name)
                    
                    temp_tile_np = np.array(temp_tile_RGB)
                    #if there is not too much blank in the image. It can cause problems when I have to calculate 
                    #eigenvalues and eigenvectors
                    if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
                        tiff.imsave(str(tile_name), temp_tile_np)
                        
        print('Saved all tiles!!')
                    
        
def normalize_tiles(path_to_tiles, reference_tile_name, beta = 0.15, alpha = 1, method = 'transparency', scan = False, standardize_bright = False):
    
    '''
    function to normalize each tile
    
    path_to_tiles: path where the tiles are stored
    reference_tile_name: name of the tile used as reference
    beta: threshold to remove transparency zones to compute the stain matrix
    alpha: percentile to use to find the 2 exreme values of eigenvectors
    method: transparency, otsu or gabor. Methods to segment blank zones that will not be used in stain matrix computation
    scan: boolean. Whether to use SCAN algorithm or not
    standardize_bright: boolean. Wheter to standardize brightness (saturate pixels under a pixel threshold)
    
    '''
    
    print('\n')
    #instantiate normalizer object
    norm = Normalizer(beta, alpha, method, scan = scan, standardize_bright = standardize_bright)
    #read reference tile to compute reference stain matrix
    img = utils.read_image(path_to_tiles + reference_tile_name)
    #fit on the reference tile
    norm.fit(img)
    
    #normalize image and extract channels
    img_norm = norm.transform(img)
    hem = norm.hematoxylin_color(img)
    eos = norm.eosin_color(img)
    hem_gray = norm.hematoxylin_gray(img)
    
    hem_gray = np.stack((hem_gray, hem_gray, hem_gray), axis = -1)
    
    #up to one folder
    file_path = Path(path_to_tiles).parent.absolute()
    #create folders
    norm_path = file_path.joinpath('Normalized')
    hem_path = file_path.joinpath('Hem')
    eos_path = file_path.joinpath('Eos')
    hem_path_gray = file_path.joinpath('Hem_gray')
    
    Path(str(norm_path)).mkdir(parents = True, exist_ok = True) 
    Path(str(hem_path)).mkdir(parents = True, exist_ok = True) 
    Path(str(eos_path)).mkdir(parents = True, exist_ok = True)
    Path(str(hem_path_gray)).mkdir(parents = True, exist_ok = True)
    
    #save in the respective folders
    tiff.imsave(str(norm_path.joinpath(reference_tile_name)), img_norm)
    tiff.imsave(str(hem_path.joinpath(reference_tile_name)), hem)
    tiff.imsave(str(eos_path.joinpath(reference_tile_name)), eos)
    tiff.imsave(str(hem_path_gray.joinpath(reference_tile_name)), hem_gray)
    
    #iterate over the folder of original tiles and normalize all the images using the fitted tile as reference
    for file in tqdm(glob.glob(path_to_tiles + '/*.tif'), desc = 'Normalizing tiles...'):
        
        img = utils.read_image(file)
        
        img_norm = norm.transform(img)
        hem = norm.hematoxylin_color(img)
        eos = norm.eosin_color(img)
        hem_gray = norm.hematoxylin_gray(img)
        
        plt.imshow(hem_gray, cmap = 'gray')
        plt.show()
        
        hem_gray = np.stack((hem_gray, hem_gray, hem_gray), axis = -1)
        
        basename = os.path.basename(file)
        
        tiff.imsave(str(norm_path.joinpath(basename)), img_norm)
        tiff.imsave(str(hem_path.joinpath(basename)), hem)
        tiff.imsave(str(eos_path.joinpath(basename)), eos)
        tiff.imsave(str(hem_path_gray.joinpath(basename)), hem_gray)
        
def segment_voronoi_otsu(path_to_tiles, path_hematoxylin_gray, sigma_spot_detection = 3, sigma_outline = 1, plot = False):
    '''
    function to normalize each tile
    
    path_to_tiles: path where the tiles are stored
    path_hematoxylin_gray: path where the hematoxylin gray images are saved
    sigma_spot_detection: first threshold to blur images for first otsu thresholding. Increasing will detect bigger nuclei
    sigma_outline: sigma for second blur
    
    
    '''
    file_path = Path(path_to_tiles).parent.absolute()
    segmentation_path = file_path.joinpath('Segmented Voronoi')
    Path(str(segmentation_path)).mkdir(parents = True, exist_ok = True)
    
    for file in tqdm(glob.glob(path_hematoxylin_gray + '/*.tif'), desc = 'Segmenting tiles with Otsu + Voronoi...'):
        
        basename = os.path.basename(file)
        
        print(file)
        
        img_org = cv.imread(path_to_tiles + basename, cv.IMREAD_COLOR)
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
        
        hem = cv.imread(file, cv.IMREAD_GRAYSCALE)
    
        input_image = np.invert(hem)
        
        if plot:
            plt.imshow(img_org, cmap='gray')
            plt.show()
            
        sigma_spot_detection = sigma_spot_detection
        sigma_outline = sigma_outline

        segmented = cle.voronoi_otsu_labeling(input_image, spot_sigma=sigma_spot_detection, 
                                              outline_sigma=sigma_outline)
        
        segmented_array = cle.pull(segmented)
        #This is a uint32 labeled image with each object given an integer value.
        segmented_array = mark_boundaries(img_org, segmented_array)
        print(segmented_array.dtype)
        print(segmented_array.max())
        tiff.imsave(str(segmentation_path.joinpath(basename)), (segmented_array * 255).astype(np.uint8))
        
def segment_MANA_watershed(path_to_tiles, path_hematoxylin_gray, threshold_max = 150, thr_small = 0.25, thr_big = 1.5, plot = False):
    '''
    function to normalize each tile
    
    path_to_tiles: path where the tiles are stored
    path_hematoxylin_gray: path where the hematoxylin gray images are saved
    threshold_max: threshold on pixel intensities to avoid black masks- Lower is more conservative
    thr_small: threshold for small area. 0.25 means that I exclude areas below area_mean*0.25
    thr_big: threshold for big area. 1.5 means that I will process big areas above area_mean*1.5
    
    
    '''
    file_path = Path(path_to_tiles).parent.absolute()
    segmentation_path = file_path.joinpath('Segmented MANA')
    Path(str(segmentation_path)).mkdir(parents = True, exist_ok = True)
    
    for file in tqdm(glob.glob(path_hematoxylin_gray + '/*.tif'), desc = 'Segmenting tiles with MANA and watershed...'):
        
        basename = os.path.basename(file)
        
        print(file)
        
        img_org = cv.imread(path_to_tiles + basename, cv.IMREAD_COLOR)
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
        
        hem = cv.imread(file, cv.IMREAD_GRAYSCALE)
    
        #input_image = np.invert(hem)
        
        if plot:
            plt.imshow(img_org, cmap='gray')
            plt.show()
            
        i = seg_ut.preprocess(hem)
        
        od = utils.RGB_to_OD(i).reshape((-1, 1))

        #set threshold
        beta = 0.15
        
        #remove all values less than a threshold. If in one of the 3 channels I have a value less than threshold
        #remove all the line (from 3,145,728 I obtain 2,685,766)
        #od_hat = od[(od > beta).any(axis=1), :]
        
        hem = hem.reshape((-1, 1))[(od > beta).any(axis=1), :]

        pwm_curve, bins_count = seg_ut.pwm(i)

        pwm_fitted = seg_ut.fit_pwm(pwm_curve, bins_count)

        thrs = seg_ut.calculate_thresholds(pwm_fitted, threshold_max = threshold_max)

        median_areas = seg_ut.compute_areas(i, thrs)

        initial_mask, initial_threshold = seg_ut.compute_initial_mask(i, thrs, median_areas)

        small_areas, normal_areas, big_areas, props = seg_ut.compute_min_max_normal_regions(initial_mask,
                                                                                     min_perc = thr_small,
                                                                                     max_perc = thr_big)

        images_ws = seg_ut.segment_big_areas(i, initial_threshold, big_areas, props)

        normal_a = seg_ut.find_normal_areas(i, props, normal_areas)

        ws = seg_ut.segment_normal_remove_small(images_ws, normal_a)
        
        segmented_array = mark_boundaries(img_org, ws)
        print(segmented_array.dtype)
        print(segmented_array.max())
        tiff.imsave(str(segmentation_path.joinpath(basename)), (segmented_array * 255).astype(np.uint8))
        
def segment_with_neural_net(path_to_tiles, path_hematoxylin_gray, model, plot = False):
    '''
    function to normalize each tile
    
    path_to_tiles: path where the tiles are stored
    path_hematoxylin_gray: path where the hematoxylin gray images are saved
    
    
    '''
    file_path = Path(path_to_tiles).parent.absolute()
    segmentation_path = file_path.joinpath('Segmented NeuralNet')
    Path(str(segmentation_path)).mkdir(parents = True, exist_ok = True)
    
    for file in tqdm(glob.glob(path_hematoxylin_gray + '/*.tif'), desc = 'Segmenting tiles with Otsu + Voronoi...'):
        
        basename = os.path.basename(file)
        
        print(file)
        
        img_org = cv.imread(path_to_tiles + basename, cv.IMREAD_COLOR)
        img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
        
        hem = cv.imread(file, cv.IMREAD_GRAYSCALE)
    
        img = np.invert(hem)
        
        if plot:
            plt.imshow(img_org, cmap='gray')
            plt.show()
            
        

        labels, _ = model.predict_instances(normalize(img))

        #This is a uint32 labeled image with each object given an integer value.
        segmented_array = mark_boundaries(img_org, labels)
        print(segmented_array.dtype)
        print(segmented_array.max())
        tiff.imsave(str(segmentation_path.joinpath(basename)), (segmented_array * 255).astype(np.uint8))

    
        
        

#Test class and functions
img_path = '/home/acina/Documents/image analysis/peso_training_wsi_1/pds_4_HE.tif'

reader = WSI_images(img_path)

mask = reader.compute_mask(level = 1)

reader.create_save_tiles('tiles_pds_4_HE', tile_size = 1024, overlap = 0, plot = True)

normalize_tiles(path_to_tiles = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/original/', reference_tile_name = '1_29.tif')

segment_voronoi_otsu(path_to_tiles = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/original/', 
                          path_hematoxylin_gray = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/Hem_gray/', 
                          sigma_spot_detection = 3, 
                          sigma_outline = 1, 
                          plot = True)

# =============================================================================
# segment_MANA_watershed(path_to_tiles = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/original/', 
#                           path_hematoxylin_gray = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/Hem_gray/', 
#                           )
# =============================================================================

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

segment_with_neural_net(path_to_tiles = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/original/', 
                          path_hematoxylin_gray = '/home/acina/Documents/image analysis/all code/tiles_pds_4_HE/Hem_gray/',
                          model = model, 
                          plot = False)