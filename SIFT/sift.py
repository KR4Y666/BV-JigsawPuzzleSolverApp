# First Version for Puzzle Solver using Sift Algorithm and OpenCV

# Imports
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image

# Read in Images  
piece_img_bgr = cv2.imread('./puzzle_data/Puzzle4_2.jpg')       # puzzle piece
template_img_bgr = cv2.imread('./puzzle_data/Puzzle4_Template.jpg')        # puzzle template
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
x_coordinate = []
y_coordinate = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt
        start.append(pt1)
        target.append(pt2)
        start_x = pt1[0]
        start_y = pt1[1]
        target_x = pt2[0]
        target_y = pt2[1]
        x_coordinate.append(target_x)
        y_coordinate.append(target_y)


mean_x = int(sum(x_coordinate)/len(x_coordinate))
mean_y = int(sum(y_coordinate)/len(y_coordinate))

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


# #show piece at its mean coordinates on the puzzle template
# piece_img_path = './puzzle_data/Puzzle4_2.jpg'
# template_img_path = './puzzle_data/Puzzle4_template.jpg'

# im1 = Image.open(piece_img_path)
# im2 = Image.open(template_img_path)
# back_im = im2.copy()
# im1 = np.array(im1)
# #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
# im1 = Image.fromarray(im1)
# im1 = im1.resize((500,500)) 
# #TODO image resizing
# back_im.paste(im1, (mean_x, mean_y))
# plt.imshow(back_im)
# plt.show()


# picture with all the matches printed 
# output final picturers       
result_img = cv2.drawMatches(piece_img_bgr,keypoints1,template_img_bgr,keypoints2,good,None,**draw_params)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.show()
