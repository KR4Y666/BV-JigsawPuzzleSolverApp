# First Version for Puzzle Solver using Sift Algorithm and OpenCV

# Imports
import cv2 
import numpy as np
from matplotlib import pyplot as plt

# Read in Images  
piece_img_bgr = cv2.imread('puzzle_data/Puzzle_Piece_Eisbaer.jpg')       # puzzle piece
template_img_bgr = cv2.imread('puzzle_data/Puzzle_Template_Eisbaer.jpg')        # puzzle template
piece_img_gray = cv2.cvtColor(piece_img_bgr, cv2.COLOR_BGR2GRAY)

#Split images in color channels
piece_img_blue = np.array(piece_img_bgr[:,:,0])
template_img_blue = np.array(template_img_bgr[:,:,0])
piece_img_green = np.array(piece_img_bgr[:,:,1])
template_img_green = np.array(template_img_bgr[:,:,1])
piece_img_red = np.array(piece_img_bgr[:,:,2])
template_img_red = np.array(template_img_bgr[:,:,2])

#Apply SIFT algorithm to a certain color channel
def sift_algorithm(piece, template, color):

    # SIFT Detector: 
    sift = cv2.SIFT_create() # initialize SIFT

    # Using sift for detecting and computing keypoints and descriptors(des)
    kp1, des1 = sift.detectAndCompute(piece,None) 
    kp2, des2 = sift.detectAndCompute(template,None)

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
    MIN_MATCH_COUNT = 10

    # get matches
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = piece_img_gray.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        template = cv2.polylines(template,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found: {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    # draw matches on canvas
    draw_params = dict(matchColor = color,
                    singlePointColor = color,
                    matchesMask = matchesMask,   # sorting out inliners
                    flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # picture with all the matches printed      
    return cv2.drawMatches(piece,kp1,template,kp2,good,None,**draw_params)

#result images after applying the sift algorithm
result_img_blue = sift_algorithm(piece_img_blue, template_img_blue, (255,0,0))
result_img_green = sift_algorithm(piece_img_green, template_img_green, (0,255,0))
result_img_red = sift_algorithm(piece_img_red, template_img_red, (0,0,255))

result_img_blue = result_img_blue[:,:,0]
result_img_green = result_img_green[:,:,1]
result_img_red = result_img_red[:,:,2]

#merge color channels
result_img = cv2.merge([result_img_blue, result_img_green, result_img_red])

#show final image (converted to RGB)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.show()
