import numpy as np
import cv2
# Libraries needed to edit/save/watch video clips
from moviepy import editor
import moviepy

# Read the video
video = cv2.VideoCapture('Waypoint_generation/Test.mp4')

# Take first frame of the video
ret, frame = video.read()

# implement on every frame
cv2.imshow('frame', frame)
cv2.waitKey(0)
# Read the image
im = cv2.imread('Waypoint_generation/Test.png')
cv2.imshow('original', im)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('blur', blur)

im_bw = cv2.threshold(blur, 135, 200, cv2.THRESH_BINARY)[1]
# cv2.imshow('bw', im_bw)

v = np.median(im_bw)
sigma = 0.33

#---- apply optimal Canny edge detection using the computed median----
lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))

# Apply Canny edge detection
canny = cv2.Canny(im_bw, lower_thresh, upper_thresh)

# Display the frame
# cv2.imshow('canny', canny)
cv2.imwrite('Waypoint_generation/canny.png', canny)
row_index,column_index = np.nonzero(canny)

mask = np.zeros(canny.shape, dtype=np.uint8)

if len(canny.shape) > 2:
        channel_count = canny.shape[2]
        ignore_mask_color = (255,) * channel_count
else:
        ignore_mask_color = 255

rows, cols = canny.shape[:2]
bottom_left  = [cols * 0.1, rows * 0.95]
top_left     = [cols * 0.5, rows * 0.8]
bottom_right = [cols * 0.9, rows * 0.95]
top_right    = [cols * 0.5, rows * 0.8]
vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_image = cv2.bitwise_and(canny, mask)
cv2.imshow('masked', masked_image)

rho = 1             
    # Angle resolution of the accumulator in radians.
theta = np.pi/180   
# Only lines that are greater than threshold will be returned.
threshold = 25      
# Line segments shorter than that are rejected.
minLineLength = 20  
# Maximum allowed gap between points on the same line to link them
maxLineGap = 500    
# function returns an array containing dimensions of straight lines 
# appearing in the input image
hough = cv2.HoughLinesP(masked_image, rho = rho, theta = theta, threshold = threshold,
                        minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
   
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
   
def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.7
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
 
     
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

result = draw_lane_lines(im, lane_lines(im, hough))

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.imwrite('Waypoint_generation/result.png', result)