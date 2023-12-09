import cv2
import numpy as np
import PIL.Image as Image

class Segment():

    def nothing(x):
        pass

    def convert_color_to_black(self, image, target_color):
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
    
    def convert_color_to_white(self, image, target_color):
        # Set a tolerance for the color conversion (in case the exact RGB values are not found)
        tolerance = 50
        
        # Define the lower and upper bounds for the target color
        lower_bound = np.array([max(value - tolerance, 0) for value in target_color])
        upper_bound = np.array([min(value + tolerance, 255) for value in target_color])

        # Create a mask for the target color
        mask = cv2.inRange(image, lower_bound, upper_bound)

        # Convert the target color to white
        result = image.copy()
        result[mask != 0] = [255, 255, 255]

        return result

    def find_surface_centroid(self,frame):
        # tl = (80, 387)
        # bl = (0, 480)
        # tr = (550, 387)
        # br = (640, 480)

        # cv2.namedWindow("Trackbars")

        # cv2.createTrackbar("L - H", "Trackbars", 0, 255, self.nothing)
        # cv2.createTrackbar("L - S", "Trackbars", 0, 255, self.nothing)
        # cv2.createTrackbar("L - V", "Trackbars", 200, 255, self.nothing)
        # cv2.createTrackbar("U - H", "Trackbars", 255, 255, self.nothing)
        # cv2.createTrackbar("U - S", "Trackbars", 50, 255, self.nothing)
        # cv2.createTrackbar("U - V", "Trackbars", 255, 255, self.nothing)
        # frame = cv2.resize(image, (640, 480))

        # pts1 = np.float32([tl, bl, tr, br])
        # pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
        # cv2.imshow('transformed_frame',transformed_frame)
        # cv2.waitKey(1000000)
        # Convert the frame to the HSV color space (Hue, Saturation, Value)
        # im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('im_gray',im_gray)
        # cv2.waitKey(1000000)
        # (thresh, im_bw) = cv2.threshold(im_gray, 100, 0, cv2.THRESH_BINARY)
        # print(thresh)
        # cv2.imshow('hsv_frame',im_bw)
        # cv2.waitKey(1000000)

        # l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        # l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        # l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        # u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        # u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        # u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # Define the lower and upper bounds for the color of the surface you want to detect
        # lower_bound = np.array([40, 90, 90])
        # upper_bound = np.array([100, 150, 150])

        # # Create a mask based on the color bounds
        # mask = cv2.inRange(frame, lower_bound, upper_bound)
        # mask = cv2.bitwise_not(mask)

        # cv2.imshow('mask',mask)
        # cv2.waitKey(1000)

        # Find contours in the mask
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assumed to be the surface)
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the centroid of the surface
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                # cv2.imshow('frame',frame)
                # cv2.waitKey(100)
                return cx, cy
            
        # Plot the centroid in the frame


        return None
    
    def read_video(self,frame):

        # cv2.imshow('frame',frame)
        # cv2.waitKey(100000)
        # Find the centroid of the surface in the current frame
        tl = (80, 387)
        bl = (0, 480)
        tr = (550, 387)
        br = (640, 480)
        # midx = np.zeros(10)
        # midy = np.zeros(10)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
        frame = Image.fromarray(frame)
        frame = frame.convert('RGB')
        frame = np.array(frame) 
        # cv2.imshow('frame',frame)
        # cv2.waitKey(100000)
        # frame = cv2.imread(frame)
        target_color = [0, 40, 0]
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image',image)   
        # cv2.waitKey(100000)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(image, matrix, (640, 480))
        # cv2.imshow('image',transformed_frame)
        # cv2.waitKey(100000)
        imb = self.convert_color_to_black(transformed_frame, target_color)
        target_color = [0,91,136]
        imw = self.convert_color_to_white(imb, target_color)
        im = cv2.cvtColor(imw, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('im',im)
        # cv2.waitKey(100000)
        ret, im = cv2.threshold(im, 127, 255, 0)
        # im = cv2.cvtColor(imw, cv2.COLOR_BGR2HSV)
        # cv2.imshow('im',im)
        # cv2.waitKey(100000)
        centroid = self.find_surface_centroid(im)
        # print(centroid)
        conversion_factory = 1./640.
        conversion_factorx = 1./480.
        if centroid:
            midy = (320-centroid[0])  * conversion_factory
            midx = (centroid[1]) * conversion_factorx
            # midp = np.array([midx,midy])
            return midx,midy
        else:
            return None,None

