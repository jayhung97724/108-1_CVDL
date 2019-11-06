#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# Make a list of calibration images
images = [cv2.imread(file) for file in glob.glob("../images/CameraCalibration/*.bmp")]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cameraMatrix = []
distCoeff = []
rvec = [] 
tvec = []

def showImages(images):
    for img in images:
        cv2.namedWindow("Show Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Show Image",img)
        cv2.waitKey(2000)
    cv2.destroyAllWindows()
# showImages(images)

def showImage(img):
    cv2.namedWindow("Show Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Show Image",img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
# showImage(images[0])

def Q1_1():
    global mtx, cameraMatrix, distCoeff, rvec, tvec, objpoints, imgpoints
    # prepare object points
    nx = 8
    ny = 11
    print('Q1.1 ...\nFinding corners ...')
    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    print('Showing images ...')
    showImages(images)
    # Calibration
    ret, cameraMatrix, distCoeff, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    np.savez('params.npz', cameraMatrix=cameraMatrix, distCoeff=distCoeff, rvec=rvec, tvec=tvec)

def Q1_2():
    print('Q1_2 ...')
    print(cameraMatrix)

def Q1_3(picIndex):
    print('Q1_3 ...')
    picIndex = picIndex-1
    rotMat,_ = cv2.Rodrigues(rvec[picIndex])
    print(np.concatenate((rotMat,tvec[picIndex]), axis=1))

def Q1_4():
    print('Q1_4 ...')
    print(distCoeff[0])

Q1_1()
# Q1_2()
# Q1_3(14)
# Q1_4()