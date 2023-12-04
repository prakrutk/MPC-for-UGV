import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image(image):
    tl = (80, 387)
    bl = (0, 480)
    tr = (550, 387)
    br = (640, 480)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

    # Matrix to warp the image for birdseye view
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(image, matrix, (640, 480))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 70, 150)  

    # Define the vertices of the dynamic ROI
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    
    # Apply ROI to the edge-detected image
    roi_edges = region_of_interest(edges, roi_vertices)
    
    # Use Hough transform to detect lines in the region of interest
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
    
    # Draw the detected lines on the original image
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)
    
    # Combine the original image with the line image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return result

# Read the video file
cap = cv2.VideoCapture('Waypoint_generation/Test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process each frame
    processed_frame = process_image(frame)
    
    # Display the processed frame
    cv2.imshow("Lane Detection", processed_frame)
    print(processed_frame.shape)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
