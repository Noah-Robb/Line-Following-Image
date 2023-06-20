import numpy as np
import cv2
from skimage.morphology import medial_axis

img = cv2.imread(r"Img Path")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

inverted_thresholded = cv2.bitwise_not(thresholded)

skeleton, dist = medial_axis(inverted_thresholded, return_distance=True)

dist_on_skel = dist * skeleton
skeleton_gray = (255 * dist_on_skel).astype(np.uint8)
            
cv2.imshow('black', img)
cv2.imshow('img', skeleton_gray)

k = cv2.waitKey(30) & 0xff
if k==27:
    cv2.destroyAllWindows()