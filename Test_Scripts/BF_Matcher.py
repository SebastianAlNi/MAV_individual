#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:33:14 2020

@author: ziemersky
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/Templates/gate_template_fixed.png', 0)
img2 = cv.imread('/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_8.png', 0)

threshold = 300
# Detect edges using Canny
#edges1 = cv.Canny(img1, threshold, threshold * 3)
#edges2 = cv.Canny(img2, threshold, threshold * 3)

orb = cv.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)# draw first 50 matches

drawing = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
#kp_img = cv.drawKeypoints(drawing, kp1, None, color=(0,255,0), flags=0)

#threshold = 100
#match_img = np.where(match_img >= threshold)
#print(kp_img)
plt.imshow(match_img)