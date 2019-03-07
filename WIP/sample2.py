from imutils import paths
import numpy as np
import imutils
import cv2 as cv
from matplotlib import pyplot as plt



cap = cv.VideoCapture(1)
ret, frame = cap.read()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# edged = cv2.Canny(gray, 35, 125)


# img = cv.imread('messi5.jpg',0)
edges = cv.Canny(frame,100,200)
plt.subplot(121),plt.imshow(frame,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
# cv.imshow('Sample Image', edged)