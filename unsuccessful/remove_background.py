import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('./puzzle_data/puzzle4_1.jpg')
# Convert to graycsale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

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
image2 = cv2.drawContours(mask, contours, -1, (0, 255, 0), 10)

plt.imshow(image2)
plt.show()