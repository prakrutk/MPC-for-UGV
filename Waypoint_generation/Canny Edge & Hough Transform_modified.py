#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np

vidcap = cv2.VideoCapture("C:\\Users\\Tirth\\OneDrive\\Desktop\\TIRTH_SHIYALA_23220_PROJECT\\Video_files\\Indian_Road_lanes.mp4")
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
    edges = cv2.Canny(mask,50,150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=130, minLineLength=50, maxLineGap=30)

    x_coords_left = []
    x_coords_right = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < 320:
                x_coords_left.extend([x1, x2])
            else:
                x_coords_right.extend([x1, x2])
            cv2.line(transformed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Calculated midpoints for both left and right lanes
    if x_coords_left and x_coords_right:
        midpoint_left = sum(x_coords_left) // len(x_coords_left)
        midpoint_right = sum(x_coords_right) // len(x_coords_right)
        avg_midpoint = (midpoint_left + midpoint_right) // 2
        cv2.circle(transformed_frame, (avg_midpoint, 240), 5, (255, 0, 0), -1)  # Draw a blue circle at the average midpoint
    else:
        avg_midpoint = 320  # Default midpoint if no lines are detected
        cv2.circle(transformed_frame, (avg_midpoint, 240), 5, (255, 0, 0), -1)  # Draw a blue circle at the default midpoint

    ## Inverse perspective transformation
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    inv_transformed_frame = cv2.warpPerspective(transformed_frame, inv_matrix, (640, 480))

    ## Blob detection in the original image
    blob_position = np.where(inv_transformed_frame[:, :, 2] == 255)  # Assuming the blob color is red
    if blob_position[0].size > 0 and blob_position[1].size > 0:
        blob_x = int(np.mean(blob_position[1]))
        blob_y = int(np.mean(blob_position[0]))
        cv2.circle(frame, (blob_x, blob_y), 5, (0, 255, 0), -1)  # Draw a green circle at the blob position

    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    msk2  = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # cv2.imshow("Original", frame)
    # cv2.imshow("Bird's Eye View", transformed_frame)
    # cv2.imshow("Lane Detection - Image Thresholding", mask)
    # cv2.imshow("Canny edge", edges)
   
    frame = cv2.resize(frame, (384, 288))
    transformed_frame = cv2.resize(transformed_frame, (384, 288))
    mask2 = cv2.resize(mask2, (384, 288))
    msk2 = cv2.resize(msk2, (384, 288))
    
    top = np.concatenate([frame, transformed_frame], axis=0)
    bottom = np.concatenate([ mask2, msk2 ], axis=0)
    final_frame = np.concatenate([ top, bottom ], axis=1)
    cv2.imshow("Views", final_frame)

    if cv2.waitKey(10) == 27:
        break

    success, image = vidcap.read()  # Read the next frame

cv2.destroyAllWindows()


# In[ ]:




