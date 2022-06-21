# Imports
from os import remove
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from remove_background import remove_background_of
import math

# given the puzzle template and the number of pieces in the template, calculate 
# the size of the single puzzle pieces
def calculate_piece_size(image, number_of_pieces):

    width, height = image.size
    area = width * height
    piece_area = area / number_of_pieces
    piece_width = int(math.sqrt(piece_area))
    piece_height = int(math.sqrt(piece_area))

    return piece_width, piece_height


# crop puzzle piece image so that the image is only as big as the puzzle piece
def crop_image(image):

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

    #crop image
    x, y = [], []

    for contour_line in contours:
        for contour in contour_line:
            x.append(contour[0][0])
            y.append(contour[0][1])

    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)

    cropped = image[y1:y2, x1:x2]

    return cropped


# apply SIFT algorithm to a certain color channel
def sift_algorithm(object, template, color):

    # SIFT Detector: 
    sift = cv2.SIFT_create() # initialize SIFT

    # using SIFT for detecting and computing keypoints (kp) and descriptors(des)
    kp1, des1 = sift.detectAndCompute(object,None) 
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
    start = []
    target = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            start.append(pt1)
            target.append(pt2)
            target_x = pt2[0]
            target_y = pt2[1]
            x_coordinate.append(target_x)
            y_coordinate.append(target_y)


    # remove outliers
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

    mean = sum(target)/len(target)
    mean_x = mean[0]
    mean_y = mean[1]

    # Set Match Treshholds
    MIN_MATCH_COUNT = 10

    # get matches
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = object.shape
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
    return mean_x, mean_y, cv2.drawMatches(object,kp1,template,kp2,good,None,**draw_params)


# apply improved SIFT algorithm
def improved_sift(object, template):

    # split object and template in color channels
    object_img_blue = np.array(object[:,:,0])
    template_img_blue = np.array(template[:,:,0])
    object_img_green = np.array(object[:,:,1])
    template_img_green = np.array(template[:,:,1])
    object_img_red = np.array(object[:,:,2])
    template_img_red = np.array(template[:,:,2])

    # apply sift algorithm on the separate color channels
    result_sift_blue = sift_algorithm(object_img_blue, template_img_blue, (255,0,0))
    result_sift_green = sift_algorithm(object_img_green, template_img_green, (0,255,0))
    result_sift_red = sift_algorithm(object_img_red, template_img_red, (0,0,255))

    result_img_blue = result_sift_blue[2]
    result_img_green = result_sift_green[2]
    result_img_red = result_sift_red[2]

    result_img_blue = result_img_blue[:,:,0]
    result_img_green = result_img_green[:,:,1]
    result_img_red = result_img_red[:,:,2]

    #calculate mean coordinates of piece in template
    mean_x_blue = result_sift_blue[0]
    mean_y_blue = result_sift_blue[1]
    mean_x_green = result_sift_green[0]
    mean_y_green = result_sift_green[1]
    mean_x_red = result_sift_red[0]
    mean_y_red = result_sift_red[1]
    mean_x = int((mean_x_blue + mean_x_green + mean_x_red)/3)
    mean_y = int((mean_y_blue + mean_y_green + mean_y_red)/3)

    return mean_x, mean_y, cv2.merge([result_img_blue, result_img_green, result_img_red])


# read in images
puzzle_piece_img_path = 'puzzle_data/puzzle4_2.jpg'
puzzle_template_img_path = 'puzzle_data/puzzle4_template.jpg'
pieces_img_path = 'puzzle_data\puzzle4_pieces.jpg'
puzzle_piece_img = crop_image(cv2.imread(puzzle_piece_img_path))   
puzzle_template_img = crop_image(cv2.imread(puzzle_template_img_path)) 
pieces_img = cv2.cvtColor(cv2.imread(pieces_img_path), cv2.COLOR_BGR2RGB)


# calculate position where a particular puzzle piece belongs to
# and move the puzzle piece to this position on the puzzle template
def search_position(piece, template, number_of_pieces):

    piece_PIL = Image.fromarray(np.asarray(piece))
    template_PIL = Image.fromarray(np.asarray(template))

    # calculate piece size
    piece_width, piece_height = calculate_piece_size(template_PIL, number_of_pieces)
    
    # calculate position of piece on template
    x_coordinate, y_coordinate, result_img = improved_sift(piece, template) 
    #print(x_coordinate, y_coordinate)

    # resize piece
    piece = piece_PIL.resize((piece_width, piece_height))

    # place piece at its right position on template
    template_copy = template_PIL.copy()
    template_copy.paste(piece, (x_coordinate, y_coordinate))

    return template_copy

plt.imshow(search_position(puzzle_piece_img, puzzle_template_img, 100))
plt.show()



# search for the puzzle piece that fits in to a particular part of the puzzle template
def search_piece(template, pieces, number_of_pieces):

    template_PIL = Image.fromarray(np.asarray(template))

    # calculate piece size
    piece_width, piece_height = calculate_piece_size(template_PIL, number_of_pieces)

    #cut out piece of template for which the puzzle piece should be found
    piece_of_template = template[0:piece_width, 0:piece_height]

    return improved_sift(piece_of_template, pieces)[2]

plt.imshow(search_piece(puzzle_template_img, pieces_img, 100))
plt.show()

