# First Version for Puzzle Solver using Sift Algorithm and OpenCV

# Imports
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math

# Read in Images  
piece_img_bgr = cv2.imread('puzzle_data/Puzzle_Piece_Eisbaer.jpg')       # puzzle piece
template_img_bgr = cv2.imread('puzzle_data/Puzzle_Template_Eisbaer.jpg')        # puzzle template
piece_img_gray = cv2.cvtColor(piece_img_bgr, cv2.COLOR_BGR2GRAY)      # puzzle piece grayscale
template_img_gray = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2GRAY)      # puzzle template grayscale

# SIFT Detector: 
sift = cv2.SIFT_create() # initialize SIFT

# Using sift for detecting and computing keypoints(kp) and descriptors(des)
keypoints1, des1 = sift.detectAndCompute(piece_img_gray,None) 
keypoints2, des2 = sift.detectAndCompute(template_img_gray,None)

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
start = []
target = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        start.append(keypoints1[m.queryIdx].pt)
        target.append(keypoints2[m.trainIdx].pt)

#print(target)

L = math.inf
n = 0
delta = 5

while L > len(target):

    target = np.array(target)

    L = len(target)

    points2 = []

    for i in range(len(target)):

        pt = target[i]
        d = (pt[0]-target[:,0])**2.+(pt[1]-target[:,1])**2.
        pts = target[d<delta**2.]

        x = np.average(pts[:,0])
        y = np.average(pts[:,1])

        points2 += [[x,y]]

    points2 = np.array(points2)
    target = np.unique(points2,axis=0)
    print(len(target))

# Set Match Treshholds
MIN_MATCH_COUNT = 5

# get matches
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = piece_img_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    template_img_gray = cv2.polylines(template_img_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found: {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

# draw matches on canvas
draw_params = dict(matchColor = None,
                   singlePointColor = None,
                   matchesMask = matchesMask,   # sorting out inliners
                   flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# picture with all the matches printed 
# output final picturers       
result_img = cv2.drawMatches(piece_img_bgr,keypoints1,template_img_bgr,keypoints2,good,None,**draw_params)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.show()
