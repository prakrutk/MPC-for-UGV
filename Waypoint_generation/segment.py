import cv2
import numpy as np

def find_surface_centroid(frame):
    # Convert the frame to the HSV color space (Hue, Saturation, Value)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color of the surface you want to detect
    lower_bound = np.array([60, 0, 130])
    upper_bound = np.array([200, 180, 200])

    # Create a mask based on the color bounds
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the surface)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the surface
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy

    return None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
input = cv2.VideoWriter('Waypoint_generation/input.mp4', fourcc, 60.0,(720,1280))

# Open a video capture object
cap = cv2.VideoCapture('Waypoint_generation/Test.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Find the centroid of the surface in the current frame
    centroid = find_surface_centroid(frame)

    if centroid:
        # Draw a circle at the centroid on the original frame
        cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

        # Display the frame with the centroid
        cv2.imshow("Surface Centroid Detection", frame)
        input.write(frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
input.release()
cap.release()
cv2.destroyAllWindows()
