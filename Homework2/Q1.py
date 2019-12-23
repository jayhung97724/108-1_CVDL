import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('imL.png',0)
imgR = cv2.imread('imR.png',0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()