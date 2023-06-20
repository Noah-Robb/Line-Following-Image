import numpy as np
import cv2
import math

pixel_Coords = []

#lower and upper values in hsv format for mask
lg = np.array([75, 52, 60])
ug = np.array([106, 255, 255])

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.dilate(eroded,kernel)
        temp = cv2.morphologyEx(temp, cv2.MORPH_DILATE, kernel)
        temp = cv2.dilate(temp,kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

img = cv2.imread(r"img Path")

h, w, c = img.shape

Xmid = int(w / 2)
Ymid = int(h / 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

inverted_thresholded = cv2.bitwise_not(thresholded)

skeleton = skeletonize(inverted_thresholded)

Line = cv2.findNonZero(skeleton)

outlier_threshold = 10  # Adjust this value based on your requirements

filtered_pixel_coords = []
prev_point = None

for pixel in Line:
    x, y = pixel[0]
    
    if prev_point is not None:
        prev_x, prev_y = prev_point
        distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        
        if distance <= outlier_threshold:
            filtered_pixel_coords.append([x, y])
    else:
        filtered_pixel_coords.append([x, y])

    prev_point = [x, y]

pixel_Coords = filtered_pixel_coords

averaged_coords = []
avg = [0, 0]
count = 0

for i, coord in enumerate(pixel_Coords):
    avg[0] += coord[0]
    avg[1] += coord[1]
    count += 1

    if count == 50 or i == len(pixel_Coords) - 1:
        avg[0] //= count
        avg[1] //= count
        averaged_coords.append(avg.copy())
        avg = [0, 0]
        count = 0

for i in range(len(averaged_coords) - 1):
    pt1 = (averaged_coords[i][0], averaged_coords[i][1])
    pt2 = (averaged_coords[i + 1][0], averaged_coords[i + 1][1])
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

for i in range(5):
    i+=1
    cv2.circle(img, (Xmid, i * 80), 5, (0, 0, 255), -1)
    
    matching = [coord for coord in pixel_Coords if coord[1] == (i * 80)]
    if len(matching) > 0:   
        cv2.line(img, (Xmid, i * 80), (matching[0][0], matching[0][1]),(0,255,0),2)

            
cv2.imshow('black', skeleton)
cv2.imshow('img', img)

k = cv2.waitKey(0) & 0xff
if k==27:
    cv2.destroyAllWindows()