import cv2
import numpy as np

def convert_color_to_white(image, target_color):
    # Set a tolerance for the color conversion (in case the exact RGB values are not found)
    tolerance = 50
    
    # Define the lower and upper bounds for the target color
    lower_bound = np.array([max(value - tolerance, 0) for value in target_color])
    upper_bound = np.array([min(value + tolerance, 255) for value in target_color])

    # Create a mask for the target color
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Convert the target color to white
    result = image.copy()
    result[mask != 0] = [0, 0, 0]

    return result

# Read the image
image = cv2.imread('transformed_frame_screenshot_05.12.2023.png')
print(image)

# Define the target color in RGB (you can change this to the specific color you want to convert)
target_color = [46, 0, 0]  # For example, this is green

# Convert the specified color to white and everything else to black
result_image = convert_color_to_white(image, target_color)

# Display the original and result images
# cv2.imshow("Original Image", image)
# cv2.imshow("Result Image", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
