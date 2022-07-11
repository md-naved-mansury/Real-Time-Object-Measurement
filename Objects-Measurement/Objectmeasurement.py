import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time
import requests

from canny_edge_detector import cannyEdgeDetector

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# This is the URL through which we are providing the image/video captured using IP Webcam
url = "http://192.168.229.158:8080/photo.jpg"

while True:

	# get function to request the data from the server
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)

	# Python cv2.imdecode() function is used to read image data from a memory cache and convert it into image format. 
	# This is generally used for loading the image efficiently from the internet.
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)

        # load the image, convert it to grayscale, and blur it slightly
	#cv2.cvtColor() method is used to convert an image from one color space to another. 
	#There are more than 150 color-space conversion methods available in OpenCV. 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Blurs an image using a Gaussian filter.
	#The function convolves the source image with the specified Gaussian kernel. In-place filtering is supported.
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        #cv2.imshow("edged",edged)
        
	# This function helps in extracting the contours from the image.
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        # loop over the contours individually
        for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 100:
                        continue

                # Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv2.minAreaRect(). 
		# It returns a Box2D structure which contains following detals - ( top-left corner(x,y), (width, height), angle of rotation ). 
		# But to draw this rectangle, we need 4 corners of the rectangle. 
		# It is obtained by the function cv2.boxPoints()
                orig = img.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding box
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        	# unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # draw lines between the midpoints
                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                        (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                        (255, 0, 255), 2)

        	# compute the Euclidean distance between the midpoints
                dB = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dA = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric
                # (in this case, inches)
                if pixelsPerMetric is None:
                        pixelsPerMetric = dB / args["width"]

        	# compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric

                # draw the object sizes on the image
                cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

                cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

                # show the output image
                cv2.imshow("Image", orig)
                time.sleep(1)
        
                k = cv2.waitKey(1)
                if k==ord('q'):
                        break
                
