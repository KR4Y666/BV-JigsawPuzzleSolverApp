# Imports
import cv2 
import numpy as np

frame = cv2.imread('puzzle_data/puzzle6_1.jpg') 
# Converting the image to grayscale.
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Using the Canny filter to get contours
edges = cv2.Canny(gray, 20, 30)
# Using the Canny filter with different parameters
edges_high_thresh = cv2.Canny(gray, 60, 120)
edges_high_thresh2 = cv2.Canny(gray, 25, 200)


# Stacking the images to print them together
# For comparison
images = np.hstack((gray, edges, edges_high_thresh, edges_high_thresh2))

# Display the resulting frame
cv2.imshow('Frame', images)

cv2.waitKey(0) 
#closing all open windows 
cv2.destroyAllWindows() 