# First Version for Puzzle Solver using Sift Algorithm and OpenCV

# Imports
import cv2 
import numpy as np
from matplotlib import pyplot as plt

# Read in Images 
img1 = cv2.imread('data/puzzle2_1.jpg',0)       # puzzle piece
img2 = cv2.imread('data/puzzle2_template.jpg',0)        # puzzle template
# -> using flag 0 in imread() method to read image as greyscale picutre

# SIFT Detector: 
sift = cv2.SIFT_create() # initialize SIFT

# Using sift for detecting and computing keypoints(kp) and descriptors(des)
kp1, des1 = sift.detectAndCompute(img1,None) 
kp2, des2 = sift.detectAndCompute(img2,None)

#FLANN -> Fast Library for Approximate Nearest Neighbors

# set index 
FLANN_INDEX_KDTREE = 1

# prepare param matches with FLANN 
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# sorting out good matches and storing them
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Set Match Treshholds
MIN_MATCH_COUNT = 5

# get matches
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

# draw matches on canvas
draw_params = dict(matchColor = (255,255,0),      # set color as (RGB)
                   singlePointColor = None,
                   matchesMask = matchesMask,   # sorting out inliners
                   flags = 2)

# picture with all the matches printed 
# output final picturers              
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()