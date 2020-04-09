#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:17:37 2020

@author: ziemersky
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

filename = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_10.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

threshold = 200
# Detect edges using Canny
edges = cv.Canny(gray, threshold, threshold * 3)

#ret, thresh = cv.threshold(gray, 127, 255, 0)
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

cv.drawContours(drawing, contours, -1, (0,255,0), 3)

plt.subplot(121),plt.imshow(edges,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(drawing)
plt.title('Contours'), plt.xticks([]), plt.yticks([])
plt.show()