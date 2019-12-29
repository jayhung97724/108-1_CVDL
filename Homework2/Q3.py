import numpy as np
import cv2
from matplotlib import pyplot as plt

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

# print(des1[index1])
# print(index2)

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

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1[index1], des2[index2], k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)

# matches = bf.match()

img3 = cv2.drawMatchesKnn(img1,sorted_kp1[:7],img2,sorted_kp2[:7],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()