#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:56:15 2020

@author: ziemersky
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def rescale(img, scale_percent):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scaled = cv.resize(img, dim, interpolation = cv2.INTER_AREA)
    return scaled

img1 = cv.imread('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/gate_template_fixed.png',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_131.png',cv.IMREAD_GRAYSCALE) # trainImage

#img1 = rescale(img1, 100)

# Initiate SIFT detector
orb = cv.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
#print(matches)
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3)
plt.show()