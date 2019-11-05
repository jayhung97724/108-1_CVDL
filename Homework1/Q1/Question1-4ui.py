#!/usr/bin/env python
# coding: utf-8
import sys
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QApplication, QMainWindow
from Q1_4ui import Ui_MainWindow
from PyQt5.uic import loadUi

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})


# ## Question 1

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
    np.savez('params.npz', cameraMatrix=cameraMatrix, distCoeff=distCoeff[0], rvec=rvec, tvec=tvec)

#     ret, cameraMatrix, distCoeff, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)

def Q1_2():
    print('Q1_2 ...')
    print(cameraMatrix)

def Q1_3(picIndex):
    print('Q1_3 ...')
    print(picIndex)
    picIndex = picIndex-1
    rotMat,_ = cv2.Rodrigues(rvec[picIndex])
    print(np.concatenate((rotMat,tvec[picIndex]), axis=1))

def Q1_4():
    print('Q1_4 ...')
    print(distCoeff[0])

# ## Question 2

# Load previously saved data
params = np.load('params.npz')
mtx = params['cameraMatrix']
dist = params['distCoeff']

# Make a list of calibration images
images2 = [cv2.imread(file) for file in glob.glob("../images/CameraCalibration/*.bmp")]

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*11,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:11].T.reshape(-1,2)

axis = np.float32([[-1,-1,0], [-1,1,0], 
                   [1,1,0], [1,-1,0], [0,0,-2]])

def draw(img, corners, imgpts):
    for i in range(0, 4):
        j = (i + 1) if i < 3 else 0
        img = cv2.line(img, tuple(imgpts[i].ravel()), tuple(imgpts[j].ravel()), (0,0,255), 3)
        img = cv2.line(img, tuple(imgpts[i].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 3)
    return img

def Q2():
    print('Q2 ...')
    for img in images2[:5]:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (8,11),None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist[0])

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img,corners2,imgpts)
            showImage(img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = loadUi('Q1_4.ui')
    w.button1_1.clicked.connect(Q1_1)
    w.button1_2.clicked.connect(Q1_2)
    w.button1_3.clicked.connect(lambda: Q1_3(int(w.comboBox.currentText())))
    w.button1_4.clicked.connect(Q1_4)
    w.button2.clicked.connect(Q2)
    w.show()
    sys.exit(app.exec_())



