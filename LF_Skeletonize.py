import numpy as np
import cv2
from skimage.morphology import medial_axis
import math

img = cv2.imread(r"C:\Users\noah.robb\OneDrive - Department for Education\History\Pictures\Camera Roll\Examples\WIN_20230617_22_11_44_Pro.jpg")

h, w, c = img.shape

Xmid = int(w / 2)
Ymid = int(h / 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 9)
_, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

inverted_thresholded = cv2.bitwise_not(thresholded)

skeleton, dist = medial_axis(inverted_thresholded, return_distance=True)

dist_on_skel = dist * skeleton
skeleton = (255 * dist_on_skel).astype(np.uint8)

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
            
cv2.imshow('black', img)
cv2.imshow('img', skeleton)

k = cv2.waitKey(0) & 0xff
if k==27:
    cv2.destroyAllWindows()