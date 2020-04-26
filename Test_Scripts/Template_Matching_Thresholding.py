#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:02:15 2020

@author: ziemersky

To Do:
    - Only allow reasonable aspect ratios:
        - Check aspect ratio at corner distribution or at extrapolation?
    - Add second, horizontal chess template
    - Sensitivity study through different ROC curves, parameters:
        - Threshold
        - Scale step
        - (min, max scale)
        - ratio bar to gate size
        - Rejection parameter m
        - Original image scale
        - Template Matching method
    - Way faster if original image is scaled down
    - Adapt polygon size, make smaller so that it is closer to the ground truth
    - Autmatically create Output folder in original image directory, if it doesn't exist yet
    
Done:
    - Filter single matches far away: reject_outliers
    - Filter matches from gates behind: Only use min/max corner matches
    - Parallelize -> global corner won't work
    
"""

import cv2 as cv
import numpy as np
from extrapolate_missing_corners import extrapolate_corners
from matplotlib import pyplot as plt
import time
#import os, psutil


#threshold = 0.96 # 0.89 for cv.TM_CCOEFF_NORMED, 0.96 for cv.TM_CCORR_NORMED
# tradeoff between falsely detected patterns and undetected true patterns
# Perhaps implement big loop that increases threshold until 4 corners are found of which one has only minimum matches
# But also remember to consider frames where one corner is hidden
# Lower threshold means more computational effort
ratio_bar_to_gate_size = 1/7.5 # relative thickness of gate bar


def rescale(img, scale):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return scaled

def draw_gate(mask, img, corners, shrink_factor = 0):
    
    pt_1 = [corners[1][0], corners[0][0]]
    pt_2 = [corners[1][1], corners[0][1]]
    pt_3 = [corners[1][2], corners[0][2]]
    pt_4 = [corners[1][3], corners[0][3]]
    
    # Shrink polygon corners to fit inner rectangle of gate
    if shrink_factor != 1:
        shrink_dist_ratio = (1 - shrink_factor) / 2
        dist_12 = abs(pt_2[0] - pt_1[0])
        dist_24 = abs(pt_4[1] - pt_2[1])
        dist_43 = abs(pt_3[0] - pt_4[0])
        dist_31 = abs(pt_1[1] - pt_3[1])
        
        pt_1[0] = pt_1[0] + dist_12 * shrink_dist_ratio * 2
        pt_1[1] = pt_1[1] + dist_31 * shrink_dist_ratio
        
        pt_2[0] = pt_2[0] - dist_12 * shrink_dist_ratio
        pt_2[1] = pt_2[1] + dist_24 * shrink_dist_ratio
        
        pt_3[0] = pt_3[0] + dist_43 * shrink_dist_ratio * 2
        pt_3[1] = pt_3[1] - dist_31 * shrink_dist_ratio
        
        pt_4[0] = pt_4[0] - dist_43 * shrink_dist_ratio
        pt_4[1] = pt_4[1] - dist_24 * shrink_dist_ratio
        
    width_gate = max(corners[0]) - min(corners[0])
    height_gate = max(corners[1]) - min(corners[1])
    mean_gate_size = (width_gate + height_gate) / 2
    bar_thickness = mean_gate_size * ratio_bar_to_gate_size
    
    line_width = int(round(bar_thickness, 0))
    if line_width == 0: line_width = 1
    
    color = (255, 255, 255)
    
    points = np.array([pt_1, pt_2, pt_4, pt_3])
    #cv.polylines(mask, np.int32([points]), True, color, line_width, lineType=4) # test whether 4 or 8 is faster and better
    
    #cv.polylines(mask, np.int32([points]), True, (0, 255, 0), 1, lineType=4)
    #cv.polylines(img, np.int32([points]), True, (0, 255, 0), 1, lineType=4)
    
    cv.fillPoly(mask, np.int32([points]), (255, 255, 255))
    
    return mask

def reject_outliers(data, m=2):
    tmp = (abs(data[0] - np.mean(data[0])) <= m * np.std(data[0])) & (abs(data[1] - np.mean(data[1])) <= m * np.std(data[1]))
    return (data[0][tmp], data[1][tmp])

def template_matching_thresholding():
    
    # Variables that are object to sensitivity studies
    match_thresh = 0.97
    step = 0.05
    img_scale = 0.7
    shrink_factor = 0.84 # 0.84 measured in original sample image

    template_name = 'Templates/chess_template8.png'
    #template_name_2 = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/chess_template5r.png'
    template = cv.imread(template_name,0)
    #template2 = cv.imread(template_name_2,0)
    
    global_corners = ([0,0,0,0],[0,0,0,0]) # Stores the coordinates of the four corners globally
    
    max_runtime = 0
    min_runtime = 10
    max_loc_runtime = 0
    time_count = 0
    times = 0
    
    for num in range(439):
        start = time.perf_counter()
    
        filename = '../../WashingtonOBRace/WashingtonOBRace/img_' + str(num) + '.png'
        
        img_rgb = cv.imread(filename)
            
        try:
            img_rgb = rescale(img_rgb, img_scale)
            img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        except:
            continue
        
        scale_max = 1 # 1.5
        scale_min = 0.4 # 0.5, do not go below 0.4 or 0.35
        scale = scale_max
        
        loc = [[],[]] # first is 0/1 = x/y, second is coordinate   
        while scale >= scale_min: # Take a look at online tutorial
            template_scaled = rescale(template, scale)
            res = cv.matchTemplate(img_gray,template_scaled,cv.TM_CCORR_NORMED) #TM_CCOEFF_NORMED
            loc_tmp = np.where(res >= match_thresh)
            
            loc[0] = np.append(loc[0], loc_tmp[0])
            loc[1] = np.append(loc[1], loc_tmp[1])
        
            scale = scale - step
            
        width, height = img_gray.shape[::-1]
        x_min = 0
        x_max = width-1
        y_min = 0
        y_max = height-1
        if len(loc[0]) > 0:
            loc = reject_outliers(loc) # use different m? E.g. m=3? Only if true matches are accidently identified as outliers
            x_min = min(loc[0])
            x_max = max(loc[0])
            y_min = min(loc[1])
            y_max = max(loc[1])
        # Check for minimum distance between max and min values, if it is too low, repeat with different threshold
        
        x_c = int(x_min + x_max) / 2
        y_c = int(y_min + y_max) / 2
        
        # Sort matches to corners
        tmp = (loc[0] <= x_c) & (loc[1] <= y_c)
        corner_1 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] <= x_c) & (loc[1] >= y_c)
        corner_2 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] >= x_c) & (loc[1] <= y_c)
        corner_3 = (loc[0][tmp], loc[1][tmp])
        
        tmp = (loc[0] >= x_c) & (loc[1] >= y_c)
        corner_4 = (loc[0][tmp], loc[1][tmp])
        
        local_corners = ([0,0,0,0],[0,0,0,0]) # Stores the coordinates of the four corners
        num_corners = 0
        
        # Calculate mean coordinates of all four corners
        # if min/max is used, the rounding can be removed
        if len(corner_1[0]) > 0:
            corner_1 = reject_outliers(corner_1)
            local_corners[0][0] = np.min(corner_1[0])
            local_corners[1][0] = np.min(corner_1[1])
            num_corners += 1
            
        if len(corner_2[0]) > 0:
            corner_2 = reject_outliers(corner_2)
            local_corners[0][1] = np.min(corner_2[0])
            local_corners[1][1] = np.max(corner_2[1])
            num_corners += 1
            
        if len(corner_3[0]) > 0:
            corner_3 = reject_outliers(corner_3)
            local_corners[0][2] = np.max(corner_3[0])
            local_corners[1][2] = np.min(corner_3[1])
            num_corners += 1
            
        if len(corner_4[0]) > 0:
            corner_4 = reject_outliers(corner_4)
            local_corners[0][3] = np.max(corner_4[0])
            local_corners[1][3] = np.max(corner_4[1])
            num_corners += 1
            
        #print(local_corners)
        # Extrapolate corners if missing
        if num_corners == 4:
            global_corners = local_corners
        if num_corners == 3 or num_corners == 2:
            global_corners = extrapolate_corners(local_corners, num_corners)
        # If only one or zero corners were found, reuse last global corners
        
        mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        
        if global_corners != ([0,0,0,0],[0,0,0,0]):
            mask = draw_gate(mask, img_rgb, global_corners, shrink_factor)
            
        end = time.perf_counter()
        
        #w, h = template.shape[::-1]
        #for pt in zip(*loc[::-1]):
            #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            #cv.circle(img_rgb, pt, 5, (0,0,255), 2)
            
        '''for pt in range(len(loc[0])):
            cv.circle(img_rgb, (int(loc[1][pt]), int(loc[0][pt])), 5, (0,0,255), 2)
        
        img_combined = cv.hconcat([img_rgb, mask])
            
        cv.imwrite('../../WashingtonOBRace/Output/img_' + str(num) + '.png',img_rgb)
        cv.imwrite('../../WashingtonOBRace/Output/comb_' + str(num) + '.png',img_combined)'''
        cv.imwrite('../../WashingtonOBRace/Output/mask_' + str(num) + '.png',mask)
        #print(round(num/438*100, 0), ' %')
        
        if (end-start) > max_runtime: max_runtime = end-start
        if (end-start) < min_runtime: min_runtime = end-start
        time_count += end-start
        times += 1
        #if (end_loc-start_loc) > max_loc_runtime: max_loc_runtime = end_loc-start_loc
    
    mean_runtime = time_count/times
    #print(f'Maximum runtime: {max_runtime:0.4f}')
    #print(f'Minimum runtime: {min_runtime:0.4f}')
    print(f'Mean runtime: {mean_runtime:0.4f}')
    #print(times)
    #print(f'Maximum local runtime: {max_loc_runtime:0.4f}')
    
    return 0

if __name__ == '__main__':
    #pid = os.getpid()
    #print(pid)
    template_matching_thresholding()

#plt.subplot(121),plt.imshow(img_rgb)
#plt.title('Matches')#, plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(mask,cmap = 'gray')
#plt.title('Mask')#, plt.xticks([]), plt.yticks([])
#plt.show()