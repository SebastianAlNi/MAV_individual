#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:02:15 2020

@author: ziemersky

To Do:
    - Filter single matches far away
    - Only allow reasonable aspect ratios:
        - Check aspect ratio at corner distribution or at extrapolation?
    - Filter matches from gates behind
    - Add second, horizontal chess template
    - Sensitivity study through different ROC curves, parameters:
        - Threshold
        - Scale step
        - (min, max scale)
        - ratio bar to gate size
    - change personal name in file directories to /../.. like in ROC curve example
"""

import cv2 as cv
import numpy as np
from extrapolate_missing_corners import extrapolate_corners
from matplotlib import pyplot as plt
import time


threshold = 0.85 # tradeoff between falsely detected patterns and undetected true patterns
# Perhaps implement big loop that increases threshold until 4 corners are found of which one has only minimum matches
# But also remember to consider frames where one corner is hidden
# Lower threshold means more computational effort
ratio_bar_to_gate_size = 1/7.5 # relative thickness of gate bar

def rescale(img, scale_percent):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return scaled

def draw_gate(mask, corners, img):
    pt_1 = [corners[1][0], corners[0][0]]
    pt_2 = [corners[1][1], corners[0][1]]
    pt_3 = [corners[1][2], corners[0][2]]
    pt_4 = [corners[1][3], corners[0][3]]
        
    width_gate = max(corners[0]) - min(corners[0])
    height_gate = max(corners[1]) - min(corners[1])
    mean_gate_size = (width_gate + height_gate) / 2
    bar_thickness = mean_gate_size * ratio_bar_to_gate_size
    
    line_width = int(round(bar_thickness, 0))
    if line_width == 0: line_width = 1
    
    color = (255, 255, 255)
    
    #cv.line(mask, pt_1, pt_2, color, line_width) # make corners sharp instead of round
    #cv.line(mask, pt_2, pt_4, color, line_width)
    #cv.line(mask, pt_4, pt_3, color, line_width)
    #cv.line(mask, pt_3, pt_1, color, line_width)
    
    points = np.array([pt_1, pt_2, pt_4, pt_3])
    cv.polylines(mask, np.int32([points]), True, color, line_width, lineType=4) # test whether 4 or 8 is faster and better
    
    cv.polylines(mask, np.int32([points]), True, (0, 255, 0), 1, lineType=4)
    cv.polylines(img, np.int32([points]), True, (0, 255, 0), 1, lineType=4)
    
    return mask

def reject_outliers(data, m=2):
    tmp = np.logical_and(abs(data[0] - np.mean(data[0])) <= m * np.std(data[0]), abs(data[1] - np.mean(data[1])) <= m * np.std(data[1]))
    return (data[0][tmp], data[1][tmp])

def template_matching_thresholding(match_thresh):

    template_name = '../../WashingtonOBRace/WashingtonOBRace/Templates/chess_template8.png'
    #template_name_2 = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/chess_template5r.png'
    template = cv.imread(template_name,0)
    #template2 = cv.imread(template_name_2,0)
    
    global_corners = ([0,0,0,0],[0,0,0,0]) # Stores the coordinates of the four corners globally
    
    runtime = 0
    
    for num in range(438):
        start = time.perf_counter()
    
        filename = '../../WashingtonOBRace/WashingtonOBRace/WashingtonOBRace/img_' + str(num+1) + '.png'
        
        img_rgb = cv.imread(filename)
            
        try:
            img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
        except:
            continue
        
        width, height = img_gray.shape[::-1]
        dist_thresh = 20 # make dependent from height and/or width by subtracting min and max coordinates
        
        scale_max = 150
        scale_min = 50
        scale = scale_max
        step = 5
        
        #res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
        #loc = np.where(res >= match_thresh)
        #loc = list(loc)
        loc = [[],[]]
        
        while scale >= scale_min: # Take a look at online tutorial
            template_scaled = rescale(template, scale)
            res = cv.matchTemplate(img_gray,template_scaled,cv.TM_CCOEFF_NORMED)
            loc_tmp = np.where( res >= match_thresh)
            #loc_tmp = list(loc_tmp)
            loc[0] = np.append(loc[0], loc_tmp[0])
            loc[1] = np.append(loc[1], loc_tmp[1])
        
            scale = scale - step
            
        #print(loc) # first is 0/1 = x/y, second is coordinate
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
        
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        corner_1 = [[], []] # top left
        corner_2 = [[], []] # top right
        corner_3 = [[], []] # bottom left
        corner_4 = [[], []] # bottom right
        
        # Sort matches to corners
        for j in range(len(loc[0])):
            if loc[0][j] <= x_c and loc[1][j] <= y_c:
                corner_1[0] = np.append(corner_1[0], loc[0][j])
                corner_1[1] = np.append(corner_1[1], loc[1][j])
                
            elif loc[0][j] <= x_c and loc[1][j] >= y_c:
                corner_2[0] = np.append(corner_2[0], loc[0][j])
                corner_2[1] = np.append(corner_2[1], loc[1][j])
                
            elif loc[0][j] >= x_c and loc[1][j] <= y_c:
                corner_3[0] = np.append(corner_3[0], loc[0][j])
                corner_3[1] = np.append(corner_3[1], loc[1][j])
                
            elif loc[0][j] >= x_c and loc[1][j] >= y_c:
                corner_4[0] = np.append(corner_4[0], loc[0][j])
                corner_4[1] = np.append(corner_4[1], loc[1][j])
                    
        '''print('Total matches: ',len(loc[0]))
        print('Corner 1 matches: ', len(corner_1[0]))
        print('Corner 2 matches: ', len(corner_2[0]))
        print('Corner 3 matches: ', len(corner_3[0]))
        print('Corner 4 matches: ', len(corner_4[0]))'''
        
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
            
        # Extrapolate corners if missing
        if num_corners == 4:
            global_corners = local_corners
        if num_corners == 3 or num_corners == 2:
            global_corners = extrapolate_corners(local_corners, num_corners)
        # If only one or zero corners were found, reuse last global corners
        
        mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        
        if global_corners != ([0,0,0,0],[0,0,0,0]):
            mask = draw_gate(mask, global_corners, img_rgb)
            
        end = time.perf_counter()
            
        #w, h = template.shape[::-1]
        #for pt in zip(*loc[::-1]):
            #cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            #cv.circle(img_rgb, pt, 5, (0,0,255), 2)
            
        for pt in range(len(loc[0])):
            cv.circle(img_rgb, (int(loc[1][pt]), int(loc[0][pt])), 5, (0,0,255), 2)
        
        img_combined = cv.hconcat([img_rgb, mask])
            
        cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/img_' + str(num+1) + '.png',img_rgb)
        cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/mask_' + str(num+1) + '.png',mask)
        cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/comb_' + str(num+1) + '.png',img_combined)
        print(round(num/438*100, 0), ' %')
        
        if (end-start) > runtime: runtime = end-start
        #print(end-start)
    
    print(f'Maximum runtime: {runtime:0.4f}')
    
    return 0

template_matching_thresholding(threshold)

#plt.subplot(121),plt.imshow(img_rgb)
#plt.title('Matches')#, plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(mask,cmap = 'gray')
#plt.title('Mask')#, plt.xticks([]), plt.yticks([])
#plt.show()