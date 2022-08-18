# Imports
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import interpolate

image = cv2.imread('puzzle_data/puzzle4_2.jpg') 

lap = cv2.Laplacian(image,cv2.CV_64F,ksize=3) 
lap = np.uint8(np.absolute(lap))
lap = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)

contour = np.zeros(lap.shape)

width, height = lap.shape

points = []

for i in range(width-1):
    for j in range(height-1):
        upper = lap[i,j]
        lower = lap[i,j+1]
        if(abs(int(upper)-int(lower)) > 50):
            contour[i,j] = 255
            points.append((i,j))
            break

for j in range(height-1):
    for i in range(width-1):
        upper = lap[i,j]
        lower = lap[i,j+1]
        if(abs(int(upper)-int(lower)) > 50):
            contour[i,j] = 255
            points.append((i,j))
            break

for i in reversed(range(width-1)):
    for j in reversed(range(height-1)):
        upper = lap[i,j]
        lower = lap[i,j+1]
        if(abs(int(upper)-int(lower)) > 50):
            contour[i,j] = 255
            points.append((i,j))
            break

for j in reversed(range(height-1)):
    for i in reversed(range(width-1)):
        upper = lap[i,j]
        lower = lap[i,j+1]
        if(abs(int(upper)-int(lower)) > 50):
            contour[i,j] = 255
            points.append((i,j))
            break


def find_nearest_point(point1, points):
    min = math.inf
    for point2 in points:
        if point1 != point2:
            distance = math.dist(point1, point2)
            if distance < min:
                nearest = point2
                min = distance
    return nearest, distance

#print(len(points))

#for point in points:
#    if find_nearest_point(point, points)[1] > 200:
#        points.remove(point)

#print(points)
#print(points[0])
#print(find_nearest_point(points[0], points))
#print(len(points))

#start = points[0]
#while len(points) > 2:
#    #print(len(points))
#    next, index = find_nearest_point(start, points)
#    cv2.line(contour, (start[1], start[0]), (next[1], next[0]), [255,255,255], 2)
#    points.remove(start)
#    start = next


mask = np.zeros_like(lap)
ret, thresh = cv2.threshold(lap, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result = cv2.drawContours(mask, contours, -1, (0,255,0), 3)


plt.imshow(result)
plt.show()