# Imports
from os import remove
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from remove_background import remove_background_of

# Read in Images  
piece_img_path = 'puzzle_data/Puzzle_Piece_Feuerwehr.jpg'
template_img_path = 'puzzle_data/Puzzle_Template_Feuerwehr.jpg'
piece_img_bgr = cv2.imread(piece_img_path)       # puzzle piece
piece_img_bgr = remove_background_of(piece_img_bgr)
template_img_bgr = cv2.imread(template_img_path) # puzzle template
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
    x_coordinate = []
    y_coordinate = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            start_x = pt1[0]
            start_y = pt1[1]
            target_x = pt2[0]
            target_y = pt2[1]
            x_coordinate.append(target_x)
            y_coordinate.append(target_y)
    mean_x = sum(x_coordinate)/len(x_coordinate)
    mean_y = sum(y_coordinate)/len(y_coordinate)

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
    return mean_x, mean_y, cv2.drawMatches(piece,kp1,template,kp2,good,None,**draw_params)

#return values of sift algorithm
result_sift_blue = sift_algorithm(piece_img_blue, template_img_blue, (255,0,0))
result_sift_green = sift_algorithm(piece_img_green, template_img_green, (0,255,0))
result_sift_red = sift_algorithm(piece_img_red, template_img_red, (0,0,255))

#result images after applying the sift algorithm
result_img_blue = result_sift_blue[2]
result_img_green = result_sift_green[2]
result_img_red = result_sift_red[2]

result_img_blue = result_img_blue[:,:,0]
result_img_green = result_img_green[:,:,1]
result_img_red = result_img_red[:,:,2]

#merge color channels
result_img = cv2.merge([result_img_blue, result_img_green, result_img_red])

#show final image (converted to RGB)
#plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#plt.show()

#calculate mean coordinates of piece in template
mean_x_blue = result_sift_blue[0]
mean_y_blue = result_sift_blue[1]
mean_x_green = result_sift_green[0]
mean_y_green = result_sift_green[1]
mean_x_red = result_sift_red[0]
mean_y_red = result_sift_red[1]
mean_x = int((mean_x_blue + mean_x_green + mean_x_red)/3)
mean_y = int((mean_y_blue + mean_y_green + mean_y_red)/3)
#print(mean_x, mean_y)

#show piece at its mean coordinates on the puzzle template
im1 = Image.open(piece_img_path)
im2 = Image.open(template_img_path)
back_im = im2.copy()
im1 = remove_background_of(np.array(im1))
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im1 = Image.fromarray(im1)
im1 = im1.resize((500,500)) #TODO image resizing
back_im.paste(im1, (mean_x, mean_y))
plt.imshow(back_im)
plt.show()

