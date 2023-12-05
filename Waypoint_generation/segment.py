import cv2
import numpy as np

class Segment():
    def find_surface_centroid(self,frame):
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
    
    def read_video(self,frame):

        # Find the centroid of the surface in the current frame
        centroid = self.find_surface_centroid(frame)
        conversion_factor = 0.001
        if centroid:
            midx = centroid[0] * conversion_factor
            midx = centroid_x * conversion_factor
            midy = centroid_y * conversion_factor
            return midx,midy
        else:
            return None

