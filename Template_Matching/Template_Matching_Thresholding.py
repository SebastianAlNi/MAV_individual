#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:02:15 2020

@author: Sebastian Nicolay

@purpose: Individual Assignment for the course AE4317 Autonomous Flight of Micro Air Vehciles
    
"""

import cv2 as cv
import numpy as np
import Supporting_Functions as sf
import time
import os

input_path = '../../WashingtonOBRace/'
output_path = '../../Output/'


"""
template_matching_thresholding
This function detects racing gates based on the template matching method and applying a matching threshold
@param none
@return none
"""
def template_matching_thresholding():
    
    # Variables that are object to sensitivity studies
    match_thresh = 0.96 # matching threshold
    step = 0.05 # template scale step size
    img_scale = 0.7 # original image scale
    shrink_factor = 0.84 # 0.84 geometrically measured in original sample image
    scale_max = 0.55 # maximum template scale
    
    # Store the coordinates of the four corners outside the loop to reuse them in difficult situations
    global_corners = ([0,0,0,0],[0,0,0,0])
    
    # Runtime measurement variables
    max_runtime = 0
    min_runtime = 10
    time_count = 0
    times = 0

    # Read template image as grayscale
    template_name = 'checked_template.png'
    template = cv.imread(template_name,0)
    
    
    '''---------------------------------------------------------
    Gate detection loop for all images from the given data set
    ---------------------------------------------------------'''
    for num in range(439):
        start = time.perf_counter() # store starting time
    
        # Read original image
        filename = input_path + 'img_' + str(num) + '.png'     
        img_rgb = cv.imread(filename)

        # Check if the image exists, otherwise skip to next image number
        try:
            # Rescale original image and create grayscale
            img_rgb = sf.rescale(img_rgb, img_scale)
            img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        except:
            continue
        
        # Define minimum and staring template scale
        scale_min = 0.4 # minimum template scale, do not go below 0.4, otherwise a lot of noise will be cetected
        scale = scale_max # starting template scale
        
        loc = [[],[]] # matrix for the detected matches, first is 0/1 = x/y, second is coordinate
        
        
        '''---------------------------------------------------------
        Template matching loop for several template scales
        ---------------------------------------------------------'''
        while scale >= scale_min: # Take a look at online tutorial
            template_scaled = sf.rescale(template, scale)
            res = cv.matchTemplate(img_gray,template_scaled,cv.TM_CCORR_NORMED) # Run template matching
            loc_tmp = np.where(res >= match_thresh) # Determine matches above the matching threshold
            
            loc[0] = np.append(loc[0], loc_tmp[0])
            loc[1] = np.append(loc[1], loc_tmp[1])
        
            scale = scale - step
            
        
        '''---------------------------------------------------------
        Determine the center of all matches
        ---------------------------------------------------------'''
        width, height = img_gray.shape[::-1]
        x_min = 0
        x_max = width-1
        y_min = 0
        y_max = height-1
        if len(loc[0]) > 0:
            loc = sf.reject_outliers(loc) # Reject matches that are far away from all other matches
            x_min = min(loc[0])
            x_max = max(loc[0])
            y_min = min(loc[1])
            y_max = max(loc[1])
        
        x_c = int(x_min + x_max) / 2
        y_c = int(y_min + y_max) / 2
        
        
        '''---------------------------------------------------------
        Assign the matches to one of the four corners with respect to the center
        ---------------------------------------------------------'''
        tmp = (loc[0] <= x_c) & (loc[1] <= y_c)
        corner_1 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] <= x_c) & (loc[1] >= y_c)
        corner_2 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] >= x_c) & (loc[1] <= y_c)
        corner_3 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] >= x_c) & (loc[1] >= y_c)
        corner_4 = (loc[0][tmp], loc[1][tmp])
        
        
        '''---------------------------------------------------------
        Reduce to one coordinate per corner by finding min/max values
        Outliers are rejected for each corner, respectively
        ---------------------------------------------------------'''        
        local_corners = ([0,0,0,0],[0,0,0,0]) # Stores the coordinates of the four corners
        num_corners = 0 # Counts the corners
        
        if len(corner_1[0]) > 0:
            corner_1 = sf.reject_outliers(corner_1)
            local_corners[0][0] = np.min(corner_1[0])
            local_corners[1][0] = np.min(corner_1[1])
            num_corners += 1
            
        if len(corner_2[0]) > 0:
            corner_2 = sf.reject_outliers(corner_2)
            local_corners[0][1] = np.min(corner_2[0])
            local_corners[1][1] = np.max(corner_2[1])
            num_corners += 1
            
        if len(corner_3[0]) > 0:
            corner_3 = sf.reject_outliers(corner_3)
            local_corners[0][2] = np.max(corner_3[0])
            local_corners[1][2] = np.min(corner_3[1])
            num_corners += 1
            
        if len(corner_4[0]) > 0:
            corner_4 = sf.reject_outliers(corner_4)
            local_corners[0][3] = np.max(corner_4[0])
            local_corners[1][3] = np.max(corner_4[1])
            num_corners += 1
            
        
        '''---------------------------------------------------------
        Extrapolation of missing corners if two or three corners were found
        ---------------------------------------------------------'''
        if num_corners == 4:
            global_corners = local_corners
        if num_corners == 3 or num_corners == 2:
            global_corners = sf.extrapolate_corners(local_corners, num_corners)
        # If only one or zero corners were found, reuse last global corners
        
        
        '''---------------------------------------------------------
        Create mask and draw gate polygon
        ---------------------------------------------------------'''
        mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        
        if global_corners != ([0,0,0,0],[0,0,0,0]):
            mask = sf.draw_gate(mask, img_rgb, global_corners, shrink_factor)
            
        end = time.perf_counter() # Store finishing time
        
        
        '''---------------------------------------------------------
        Draw matches for visualization and store images
        ---------------------------------------------------------'''
        for pt in range(len(loc[0])):
            cv.circle(img_rgb, (int(loc[1][pt]), int(loc[0][pt])), 5, (0,0,255), 2)
        
        img_combined = cv.hconcat([img_rgb, mask])
            
        cv.imwrite(output_path + 'img_' + str(num) + '.png',img_rgb)
        cv.imwrite(output_path + 'comb_' + str(num) + '.png',img_combined)
        cv.imwrite(output_path + 'mask_' + str(num) + '.png',mask)
        
        # Calculate runtime
        if (end-start) > max_runtime: max_runtime = end-start
        if (end-start) < min_runtime: min_runtime = end-start
        time_count += end-start
        times += 1
    
    mean_runtime = time_count/times
    print(f'Maximum runtime: {max_runtime:0.4f}')
    print(f'Minimum runtime: {min_runtime:0.4f}')
    print(f'Mean runtime: {mean_runtime:0.4f}')
    
    return 0

if __name__ == '__main__':
    if os.path.isdir(output_path):
        print('Output directory already exists: %s' % output_path)
    else:
        try:
            os.mkdir(output_path)
        except OSError:
            print('Creation of the directory %s failed' % output_path)
        else:
            print('Successfully created the directory %s ' % output_path)
    if os.path.isdir(input_path):
        template_matching_thresholding()
    else:
        print('Input folder not found. Please place the WashingtonOBRace folder including the original images next to the MAV_individual folder. Do not use subfolders.')