import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('Aerial1.jpg',0) # queryImage
img2 = cv2.imread('Aerial2.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create(6)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Sort them in the order of their distance.
sorted_good = sorted(good, key = lambda x:x[0].distance)

img1 = cv2.drawKeypoints(img1, kp1, img1, flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('FeatureAerial1.jpg', img1)
cv2.imshow('FeatureAerial1.jpg', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
img2 = cv2.drawKeypoints(img2, kp2, img2, flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('FeatureAerial2.jpg', img2)
cv2.imshow('FeatureAerial2.jpg', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,sorted_good[7:15],None,flags=2)

plt.imshow(img3),plt.show()
