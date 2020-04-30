#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:10:25 2020

@author: Sebastian Nicolay

@purpose: Supporing functions for the Individual Assignment for the course AE4317 Autonomous Flight of Micro Air Vehciles
"""

import cv2 as cv
import numpy as np

"""
rescale
Rescales an image to a different resolution
@param img - image object, OpenCV data type
@param scale - desired scale, where 1 has no effect
@return scaled image object, OpenCV data type
"""
def rescale(img, scale):
    # Determine new image width and height
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return scaled


"""
reject_outliers
Rejects matches that have a distance of two times the standard deviation to the mean coordinates
@param data - matrix with two columns for the x and y coordinates
@param m - factor to be multiplied with the standard deviation for classifying matches as outliers
@return data matrix without outliers
"""
def reject_outliers(data, m=2):
    # Determine data points that are within m * the standard variation
    tmp = (abs(data[0] - np.mean(data[0])) <= m * np.std(data[0])) & (abs(data[1] - np.mean(data[1])) <= m * np.std(data[1]))
    return (data[0][tmp], data[1][tmp])


"""
extrapolate_corners
This function extrapolates missing corners based on the coordinates that are closest to the missing corner
It works if three corners are detected or if two corners aligned diagonally are detected
@param corners - tuple object that contains x and y coordinates of all corners
@param num - number of successfully detected corners
@return tuple object that contains x and y coordinates of all corners including the extrapolated corners
"""
def extrapolate_corners(corners, num):
    if num == 3 or num == 2:
        # If one or two corners are missing, estimate coordinates based on given corners
        if corners[0][0] == 0 and corners[1][0] == 0: # Check if the corner is missing
            corners[0][0] = corners[0][1] # Copy respective coordinates from the corner closest to the missing corner
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
        
    return corners


"""
draw_gate
Draws the detected gate on the mask image
    either the actual gate or the gate's opening polygon
@param mask - mask image object, OpenCV data type
@param img - original image object, OpenCV data type
@param corners - tuple object that contains the x and y coordinates of the four corners
@param shrink factor - factor between corner pattern polygon and gate opening polygon
@return mask image object, OpenCV data type
"""
def draw_gate(mask, img, corners, shrink_factor = 0):
    # Copy coordinates of all points
    pt_1 = [corners[1][0], corners[0][0]]
    pt_2 = [corners[1][1], corners[0][1]]
    pt_3 = [corners[1][2], corners[0][2]]
    pt_4 = [corners[1][3], corners[0][3]]
    
    # Shrink polygon corners to fit inner opening of gate
    if shrink_factor != 1:
        shrink_dist_ratio = (1 - shrink_factor) / 2 # distance ratio to be added or subtracted from the respective coordinates
        # Determine distances od the detected corners
        dist_12 = abs(pt_2[0] - pt_1[0])
        dist_24 = abs(pt_4[1] - pt_2[1])
        dist_43 = abs(pt_3[0] - pt_4[0])
        dist_31 = abs(pt_1[1] - pt_3[1])
        
        # Add or subtract the respective distance ratios
        # Multiply by two for left corners since template matching coordinates are always given on the left side of the match
        pt_1[0] = pt_1[0] + dist_12 * shrink_dist_ratio * 2
        pt_1[1] = pt_1[1] + dist_31 * shrink_dist_ratio
        
        pt_2[0] = pt_2[0] - dist_12 * shrink_dist_ratio
        pt_2[1] = pt_2[1] + dist_24 * shrink_dist_ratio
        
        pt_3[0] = pt_3[0] + dist_43 * shrink_dist_ratio * 2
        pt_3[1] = pt_3[1] - dist_31 * shrink_dist_ratio
        
        pt_4[0] = pt_4[0] - dist_43 * shrink_dist_ratio
        pt_4[1] = pt_4[1] - dist_24 * shrink_dist_ratio
    
    # Determine gate size and assumed thickness of the gate's bars
    # This block is only important if the actual gate is drawn and not the inner polygon
    width_gate = max(corners[0]) - min(corners[0])
    height_gate = max(corners[1]) - min(corners[1])
    mean_gate_size = (width_gate + height_gate) / 2
    ratio_bar_to_gate_size = 1/7.5 # relative thickness of gate bar, measured geometrically
    bar_thickness = mean_gate_size * ratio_bar_to_gate_size
    
    line_width = int(round(bar_thickness, 0))
    if line_width == 0: line_width = 1
    
    #color = (255, 255, 255)
    
    points = np.array([pt_1, pt_2, pt_4, pt_3])
    # Draw a thick polyline which resembles the bars of the gate
    #cv.polylines(mask, np.int32([points]), True, color, line_width, lineType=4)
        
    # Draw a filled white polygon at the opening of the gate
    cv.fillPoly(mask, np.int32([points]), (255, 255, 255))
    
    return mask