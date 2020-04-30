#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:10:25 2020

@author: Sebastian Nicolay
"""

"""
Function Description
Input:
    
Return:

"""
def extrapolate_corners(corners, num):
    if num == 3 or num == 2:
        # If one corner is missing, estimate coordinates based on given corners
        if corners[0][0] == 0 and corners[1][0] == 0:
            corners[0][0] = corners[0][1]
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
            
    #elif num == 2: # Add function for two missing corners
    #    corners
        
    #else:
    #    corners =  ([0,0,0,0],[0,0,0,0])
        
    return corners