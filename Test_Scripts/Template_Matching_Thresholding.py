#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:02:15 2020

@author: ziemersky
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

filename = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_20.png'
img_rgb = cv.imread(filename)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/chess_template2.png',0)

dist_thresh = 20
match_thresh = 0.75

i = 95
step = 5

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
loc = np.where( res >= match_thresh)
loc = list(loc)

while i >= 50:
    template_scaled = rescale(template, i)
    res = cv.matchTemplate(img_gray,template_scaled,cv.TM_CCOEFF_NORMED)
    loc_tmp = np.where( res >= match_thresh)
    loc_tmp = list(loc_tmp)
    loc[0] = np.append(loc[0], loc_tmp[0])
    loc[1] = np.append(loc[1], loc_tmp[1])

    i = i - step
    
print(loc) # first is 0/1 = x/y, second is coordinate
    
w, h = template.shape[::-1]
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    

#plt.subplot(121),plt.imshow(res,cmap = 'gray')
#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img_rgb,cmap = 'gray')
#plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#plt.show()
plt.imshow(img_rgb)