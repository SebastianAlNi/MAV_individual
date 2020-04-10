#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:02:15 2020

@author: ziemersky

To Do:
    - Filter single matches far away
    - Only allow reasonable aspect ratios
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def rescale(img, scale_percent):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return scaled


template_name = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/chess_template2.png'
template = cv.imread(template_name,0)

ratio_bar_to_gate_size = 1/7.5 # relative thickness of gate bar

for num in range(438):

    filename = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_' + str(num+1) + '.png'
    
    img_rgb = cv.imread(filename)
        
    try:
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    except:
        #print('File ', num+1, 'not found.')
        continue
    
    width, height = img_gray.shape[::-1]
    dist_thresh = 20 # make dependent from height and/or width by subtracting min and max coordinates
    match_thresh = 0.75 # tradeoff between falsely detected patterns and undetected true patterns
    # Perhaps implement big loop that increases threshold until 4 corners are found of which one has only minimum matches
    # But also remember to consider frames where one corner is hidden
    # Lower threshold means more computational effort
    i = 150
    step = 2
    
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= match_thresh)
    loc = list(loc)
    
    while i >= 40: # Take a look at online tutorial
        template_scaled = rescale(template, i)
        res = cv.matchTemplate(img_gray,template_scaled,cv.TM_CCOEFF_NORMED)
        loc_tmp = np.where( res >= match_thresh)
        loc_tmp = list(loc_tmp)
        loc[0] = np.append(loc[0], loc_tmp[0])
        loc[1] = np.append(loc[1], loc_tmp[1])
    
        i = i - step
        
    #print(loc) # first is 0/1 = x/y, second is coordinate
    x_min = 0
    x_max = width-1
    y_min = 0
    y_max = height-1
    if len(loc[0]) > 0:
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
    
    corners = ([0,0,0,0],[0,0,0,0]) # Stores the coordinates of the four corners
    
    # Calculate mean coordinates of all four corners
    if len(corner_1[0]) > 0:
        corners[0][0] = int(round(np.mean(corner_1[0]), 0))
        corners[1][0] = int(round(np.mean(corner_1[1]), 0))
        
    if len(corner_2[0]) > 0:
        corners[0][1] = int(round(np.mean(corner_2[0]), 0))
        corners[1][1] = int(round(np.mean(corner_2[1]), 0))
        
    if len(corner_3[0]) > 0:
        corners[0][2] = int(round(np.mean(corner_3[0]), 0))
        corners[1][2] = int(round(np.mean(corner_3[1]), 0))
        
    if len(corner_4[0]) > 0:
        corners[0][3] = int(round(np.mean(corner_4[0]), 0))
        corners[1][3] = int(round(np.mean(corner_4[1]), 0))
        
    # If one corner is missing, estimate coordinates based on given corners
    if corners[0][0] == 0 and corners[1][0] == 0:
        corners[0][0] = corners[0][1]
        corners[1][0] = corners[1][2]
        
    if corners[0][1] == 0 and corners[1][1] == 0:
        corners[0][1] = corners[0][0]
        corners[1][1] = corners[1][3]
        
    if corners[0][2] == 0 and corners[1][2] == 0:
        corners[0][2] = corners[0][3]
        corners[1][2] = corners[1][0]
        
    if corners[0][3] == 0 and corners[1][3] == 0:
        corners[0][3] = corners[0][2]
        corners[1][3] = corners[1][1]
    
    
    '''print('Corner 1: ', corners[0][0], ', ', corners[1][0])
    print('Corner 2: ', corners[0][1], ', ', corners[1][1])
    print('Corner 3: ', corners[0][2], ', ', corners[1][2])
    print('Corner 4: ', corners[0][3], ', ', corners[1][3])'''
    
    mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    
    pt_1 = (corners[1][0], corners[0][0])
    pt_2 = (corners[1][1], corners[0][1])
    pt_3 = (corners[1][2], corners[0][2])
    pt_4 = (corners[1][3], corners[0][3])
        
    width_gate = max(corners[0]) - min(corners[0])
    height_gate = max(corners[1]) - min(corners[1])
    mean_gate_size = (width_gate + height_gate) / 2
    bar_thickness = mean_gate_size * ratio_bar_to_gate_size
    
    line_width = int(round(bar_thickness, 0))
    if line_width == 0: line_width = 1
    
    cv.line(mask, pt_1, pt_2, (255, 255, 255), line_width) # make corners sharp instead of round
    cv.line(mask, pt_2, pt_4, (255, 255, 255), line_width)
    cv.line(mask, pt_4, pt_3, (255, 255, 255), line_width)
    cv.line(mask, pt_3, pt_1, (255, 255, 255), line_width)
        
    w, h = template.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        
    img_combined = cv.hconcat([img_rgb, mask])
        
    cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/img_' + str(num+1) + '.png',img_rgb)
    cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/mask_' + str(num+1) + '.png',mask)
    cv.imwrite('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Output/comb_' + str(num+1) + '.png',img_combined)
    #print(round(num/438*100, 0), ' %')

#plt.subplot(121),plt.imshow(img_rgb)
#plt.title('Matches')#, plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(mask,cmap = 'gray')
#plt.title('Mask')#, plt.xticks([]), plt.yticks([])
#plt.show()