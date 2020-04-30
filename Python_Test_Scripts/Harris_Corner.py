#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:32:22 2020

@author: ziemersky
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/ziemersky/Documents/Autonomous_Flight_of_Micro_Air_Vehicles/Individual Assignment/WashingtonOBRace/WashingtonOBRace/img_25.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

#dst = cv2.dilate(dst, None)

img[dst>0.01*dst.max()] = [0, 0, 255]
print(dst)
plt.imshow(img)

#cv2.imshow('dst', img)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()