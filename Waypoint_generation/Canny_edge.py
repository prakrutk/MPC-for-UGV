# Read a video stream from camera (here, a video file) and detect edges using Canny edge detection algorithm

import cv2
import numpy as np

# Read the image
im = cv2.imread('Waypoint_generation/Test.png')
cv2.imshow('original', im)
cv2.waitKey(0)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
canny = cv2.Canny(blur, 50, 150)

# Display the frame
cv2.imshow('canny', canny)
cv2.waitKey(0)

