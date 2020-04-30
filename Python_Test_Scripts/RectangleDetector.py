#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:37:21 2020

@author: ziemersky
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
font = cv2.FONT_HERSHEY_COMPLEX

def rescale(img, scale_percent):
    # Scale image resolution
    #scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    scaled = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return scaled

img = cv2.imread("/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_57.png", cv2.IMREAD_GRAYSCALE)
img = rescale(img, 50)
#_, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
threshold = cv2.Canny(img, 20, 50)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        cv2.drawContours(img, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    #if len(approx) == 3:
    #    cv2.putText(img, "Triangle", (x, y), font, 1, (0))
    #if len(approx) == 4:
        #cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
    '''elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
    elif 6 < len(approx) < 15:
        cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (0))'''
        
#cv2.imshow("shapes", img)
#cv2.imshow("Threshold", threshold)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        
plt.subplot(121),plt.imshow(threshold, cmap='gray')
plt.title('Matches')#, plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img, cmap='gray')
plt.title('Mask')#, plt.xticks([]), plt.yticks([])
plt.show()