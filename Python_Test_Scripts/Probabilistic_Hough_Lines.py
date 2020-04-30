#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:11:59 2020

@author: ziemersky
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

filename = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_60.png'
img = cv.imread(filename)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,300,900,apertureSize = 3)

threshold = 30
lines = cv.HoughLinesP(edges,1,np.pi/180,threshold,minLineLength=50,maxLineGap=10)
#print(lines)
#lines = cv.HoughLines(edges,1,np.pi/180,200)
drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(drawing,(x1,y1),(x2,y2),(0,255,0),2)

plt.subplot(121),plt.imshow(edges,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(drawing)
plt.title('Hough Lines'), plt.xticks([]), plt.yticks([])
plt.show()