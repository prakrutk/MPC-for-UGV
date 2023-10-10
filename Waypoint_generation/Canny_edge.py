# Read a video stream from camera (here, a video file) and detect ground surface using Canny edge detection

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
cv2.imwrite('Waypoint_generation/canny.png', canny)
cv2.waitKey(0)

# Find contours
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
cv2.imshow('contours', im)
cv2.imwrite('Waypoint_generation/contours.png', im)
cv2.waitKey(0)
