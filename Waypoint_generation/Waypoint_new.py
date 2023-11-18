#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np

vidcap = cv2.VideoCapture("Waypoint_generation/Test.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while success:
    frame = cv2.resize(image, (640, 480))

    ## Choosing points for perspective transformation
    tl = (80, 387)
    bl = (0, 480)
    tr = (550, 387)
    br = (640, 480)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Matrix to warp the image for birdseye view
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    ### Hough Transform for Lane Detection
    gray_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=130, minLineLength=50, maxLineGap=30)

    x_coords = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(transformed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x_coords.extend([x1, x2])

    if x_coords:
        midpoint = sum(x_coords) // len(x_coords)
        cv2.circle(transformed_frame, (midpoint, 240), 5, (255, 0, 0), -1)  # Draw a blue circle at the midpoint
    else:
        midpoint = 320  # Default midpoint if no lines are detected
        cv2.circle(transformed_frame, (midpoint, 240), 5, (255, 0, 0), -1)  # Draw a blue circle at the default midpoint


    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)

    if cv2.waitKey(10) == 27:
        break

    success, image = vidcap.read()  # Read the next frame

cv2.destroyAllWindows()
