import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow
from hw2ui import Ui_MainWindow
from PyQt5.uic import loadUi

def Q1():
    imgL = cv2.imread('imL.png',0)
    imgR = cv2.imread('imR.png',0)

    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

def Q2():
    img_rgb = cv2.imread('ncc_img.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('ncc_template.jpg',0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    res_COR = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
    result = cv2.normalize(res_COR, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    threshold = 0.95
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imwrite('res.png', img_rgb)
    cv2.imshow('CCORR_NORMED', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imshow('CCORR_NORMED', res_COR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imshow('res.png', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Q3_1():
    img1 = cv2.imread('Aerial1.jpg',0) # queryImage
    img2 = cv2.imread('Aerial2.jpg',0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    sorted_kp1 = sorted(kp1, key = lambda x:x.size, reverse=True)
    sorted_kp2 = sorted(kp2, key = lambda x:x.size, reverse=True)

    img1 = cv2.drawKeypoints(img1, sorted_kp1[:7], img1, flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('FeatureAerial1.jpg', img1)
    cv2.imshow('FeatureAerial1.jpg', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img2 = cv2.drawKeypoints(img2, sorted_kp2[:7], img2, flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('FeatureAerial2.jpg', img2)
    cv2.imshow('FeatureAerial2.jpg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Q3_2():
    img1 = cv2.imread('Aerial1.jpg',0) # queryImage
    img2 = cv2.imread('Aerial2.jpg',0) # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    sorted_kp1 = sorted(kp1, key = lambda x:x.size, reverse=True)
    sorted_kp2 = sorted(kp2, key = lambda x:x.size, reverse=True)

    index1 = []
    index2 = []

    for i in range(7):
        index1.append(kp1.index(sorted_kp1[i]))
        index2.append(kp2.index(sorted_kp2[i]))

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1[index1], des2[index2], k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1,sorted_kp1[:7],img2,sorted_kp2[:7],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = loadUi('hw2.ui')
    w.pushButton_Q1.clicked.connect(Q1)
    w.pushButton_Q2.clicked.connect(Q2)
    w.pushButton_Q3_1.clicked.connect(Q3_1)
    w.pushButton_Q3_2.clicked.connect(Q3_2)

    w.show()
    sys.exit(app.exec_())