#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:47:33 2020

@author: sebastian
"""

from collections import namedtuple
import numpy as np
import cv2 as cv
import csv
from matplotlib import pyplot as plt

file_path = '../../WashingtonOBRace/WashingtonOBRace/'

# define the `Detection` object
gt_object = namedtuple("gt_object", ["image", "gt"])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_ground_truth_points(img_name):
    ground_truth = ([0,0,0,0],[0,0,0,0])
    data = []
    gt_csv_file_path = file_path + 'corners_mod.csv'
    
    with open(gt_csv_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
                
    col = [x[0] for x in data]

    if img_name in col:
        for x in range(len(data)):
            if img_name == data[x][0]:
                for i in range(4):
                    ground_truth[0][i] = int(data[x][2*(i+1)])
                    ground_truth[1][i] = int(data[x][2*i+1])
                break

    return ground_truth

def get_ground_truth_mask(width, height, gt_img_name):
    gt_points = get_ground_truth_points(img_name)
    pt_1 = [gt_points[1][0], gt_points[0][0]]
    pt_2 = [gt_points[1][1], gt_points[0][1]]
    pt_3 = [gt_points[1][2], gt_points[0][2]]
    pt_4 = [gt_points[1][3], gt_points[0][3]]
    points = np.array([pt_1, pt_2, pt_3, pt_4])
    mask = np.zeros((width, height, 3), dtype=np.uint8)
    cv.fillPoly(mask, np.int32([points]), (255, 255, 255))
    return mask

for num in range(439):
    img_name = 'img_' + str(num) + '.png'
    img_filepath = file_path + img_name
    
    try:
        img_rgb = cv.imread(img_filepath)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    except:
        continue
    
    mask_img = get_ground_truth_mask(img_gray.shape[0], img_gray.shape[1], img_name)
    cv.imwrite(file_path + 'Ground_Truth_Polygons/' + img_name, mask_img)
    #plt.imshow(mask_img, cmap = 'gray')