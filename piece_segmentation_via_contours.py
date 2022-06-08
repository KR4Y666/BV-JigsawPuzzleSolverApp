import numpy as np
import cv2
from matplotlib import pyplot as plt

# read the image
image = cv2.imread("puzzle_data\Puzzle_Piece_Feuerwehr.jpg")

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create binary image and apply morphology operator (closing)
ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
bg = cv2.dilate(closing, kernel, iterations = 1)
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
ret, fg = cv2.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)
fg = fg.astype(np.uint8)

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
mask = np.zeros_like(image)
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 10)

#crop image
x, y = [], []

for contour_line in contours:
    for contour in contour_line:
        x.append(contour[0][0])
        y.append(contour[0][1])

x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

cropped = image[y1:y2, x1:x2]

# show cropped image with the drawn contours
plt.imshow(cropped)
plt.show()

