#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:00:13 2020

@author: ziemersky
"""

import cv2 as cv
import numpy as np

def rescale(img, scale):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return scaled

def scale_masks(scale):
    for num in range(438):
        filename = '../../WashingtonOBRace/WashingtonOBRace/Ground_Truth_Polygons/img_' + str(num+1) + '.png'
        try:
            mask = cv.imread(filename)
            mask = rescale(mask, scale)
        except:
            continue    
        
        out_file = '../../WashingtonOBRace/WashingtonOBRace/Scaled_Masks/mask_' + str(num+1) + '.png'
        cv.imwrite(out_file,mask)
        
    return 0

scale_masks(0.7)