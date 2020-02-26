# Author: Policarpo Baquera
# 64761 Experimental Capture
# Carnegie Mellon University 
# Spring 2020
# Based on OpenCV documentation
# https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findcirclesgrid

import cv2
import numpy as np
import os

# Target and Source images (the images should be in the same folder as the codefile)
imgTarget = cv2.imread('target.jpg', cv2.IMREAD_COLOR)
imgSource = cv2.imread('source.jpg', cv2.IMREAD_COLOR)

# CirclesGrid searching algorithm
patternSize = (4, 11)
criteria = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
(retval1, centers1) = cv2.findCirclesGrid(imgTarget, patternSize, flags = criteria)
(retval2, centers2) = cv2.findCirclesGrid(imgSource, patternSize, flags = criteria)

# Draws and displays the centers of the circles
img1 = cv2.drawChessboardCorners(imgTarget.copy(), (4, 11), centers1, retval1)
cv2.imshow('target',img1)
cv2.waitKey(500)
cv2.destroyAllWindows()
img2 = cv2.drawChessboardCorners(imgSource.copy(), (4, 11), centers2, retval2)
cv2.imshow('source',img2)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Homography Matrix (h)
h, mask = cv2.findHomography(centers2, centers1, cv2.RANSAC)

# Stores the Homography Matrix in a text file
matrix = open('matrix.txt', 'w')
matrix.write(f'{list(h)}')
matrix.close()

# Image warping (Source image adapts Target image's perspective)
height, width, channels = imgTarget.shape
result = cv2.warpPerspective(imgSource, h, (width, height))
cv2.imwrite('result.jpg', result)
cv2.imshow('result', result)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Computes the transformation of the images stored in the 'source' folder
pathSource = os.getcwd() + '\\source'
pathFinal = os.getcwd() + '\\warped'
print('\nWriting.....................................................\n')
for pathImage in os.listdir(pathSource):
    # Image transformation
    image = cv2.imread(f'{pathSource}\\{pathImage}', cv2.IMREAD_COLOR)
    result = cv2.warpPerspective(image, h, (width, height))
    # Image storing (adapted to specific data naming)
    name = str(pathImage)
    number = name.split(' ')[1]
    # Image storing
    cv2.imwrite(f'{pathFinal}\\warped {number}', result)
    print(f'warped {number}')
print('\nDone!')