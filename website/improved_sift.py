#! C:\Python\Python310\python.exe

# imports
import cv2 
import numpy as np
from PIL import Image


# given the puzzle template and the number of horizontal and vertical pieces in the template, 
# calculate the size of the single puzzle pieces
def calculate_piece_size(image, number_of_pieces_hor, number_of_pieces_ver):

    width, height = image.size
    piece_width = int(width/number_of_pieces_hor)
    piece_height = int(height/number_of_pieces_ver)

    return piece_width, piece_height


# crop puzzle piece image so that the image is only as big as the puzzle piece
def crop_image(image):

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # create binary image and apply morphology operator (closing)
    _, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel, iterations = 15)
    distance_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    _, fg = cv2.threshold(distance_transform, 0.02*distance_transform.max(), 255, 0)
    fg = fg.astype(np.uint8)

    # find the contours from the thresholded image
    contours,_ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw all contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 10)

    #crop image
    x, y = [], []

    for contour_line in contours:
        for contour in contour_line:
            x.append(contour[0][0])
            y.append(contour[0][1])

    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    cropped = image[y1:y2, x1:x2]

    return cropped


# remove points that differ from all other points
def remove_outliers(points):

    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]

    # compute the arithmetic mean of all x- and y-points
    mean_x = np.mean(x_points)
    mean_y = np.mean(y_points)

    # compute the standard deviation of all x- and y-points 
    standard_deviation_x = np.std(x_points)
    standard_deviation_y = np.std(y_points)

    # compute distance from mean for all x- and y-points
    distance_from_mean_x = abs(x_points - mean_x)
    distance_from_mean_y = abs(y_points - mean_y)

    max_deviations = 2

    # search for all x- and y-points which aren't outliers
    not_outlier_x = distance_from_mean_x < max_deviations * standard_deviation_x
    not_outlier_y = distance_from_mean_y < max_deviations * standard_deviation_y

    # save all "not-outlier-points" in array
    result = []
    for i in range(len(not_outlier_x)):
         if not_outlier_x[i] and not_outlier_y[i]:
            result.append([x_points[i], y_points[i]])
    
    return result

# apply SIFT algorithm to a certain color channel
def sift_algorithm(object, template, color):

    # initialize SIFT
    sift = cv2.SIFT_create()

    # use SIFT to detect and compute keypoints (kp) and descriptors(des)
    object_kp, object_des = sift.detectAndCompute(object,None) 
    templ_kp, templ_des = sift.detectAndCompute(template,None)

    # use FLANN for fast nearest neighbors search 
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # get two best matches
    matches = flann.knnMatch(object_des,templ_des,k=2)

    # sort out bad matches
    best_matches = []
    x_coordinate = []
    y_coordinate = []
    start = []
    target = []
    for x,y in matches:
        if 0.75*y.distance > x.distance:
            pt1 = object_kp[x.queryIdx].pt
            pt2 = templ_kp[x.trainIdx].pt
            start.append(pt1)
            target.append(pt2)
            target_x = pt2[0]
            target_y = pt2[1]
            x_coordinate.append(target_x)
            y_coordinate.append(target_y)
            best_matches.append(x)


    # remove outliers from both images 
    start = remove_outliers(start)
    target = remove_outliers(target)

    # calculate middle point of all found matches in start image
    # (len(x_start) and len(y_start) have to be != 0)
    x_start = [point[0] for point in start]
    y_start = [point[1] for point in start]
    if(len(x_start) != 0):
        mean_start_x = sum(x_start)/len(x_start)
    else:
        mean_start_x = 0
    if(len(y_start) != 0):
        mean_start_y = sum(y_start)/len(y_start)
    else:
        mean_start_y = 0
    
    # calculate middle point of all found matches in target image
    # (len(x_target) and len(y_target) have to be != 0)
    x_target = [point[0] for point in target]
    y_target = [point[1] for point in target]
    if(len(x_target) != 0):
        mean_target_x = sum(x_target)/len(x_target)
    else:
        mean_target_x = 0
    if(len(y_target) != 0):
        mean_target_y = sum(y_target)/len(y_target)
    else:
        mean_target_y = 0

    # get matches
    height, width = object.shape
    source_points = np.float32([object_kp[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    dest_points = np.float32([templ_kp[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    homography_matrix, mask = cv2.findHomography(source_points, dest_points, cv2.RANSAC,5.0)
    matches_mask = mask.ravel().tolist()
    points = np.float32([[0,0],[0,height-1],[width-1,height-1],[width-1,0]]).reshape(-1,1,2)
    dest = cv2.perspectiveTransform(points,homography_matrix)
    template = cv2.polylines(template,[np.int32(dest)],True,(0,255,0),15, cv2.LINE_AA)

    # draw matches on canvas
    draw_matches = dict(matchColor = color, singlePointColor = color, matchesMask = matches_mask, flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # picture with all the matches printed
    match_img = cv2.drawMatches(object, object_kp, template, templ_kp, best_matches, None, **draw_matches)

    return mean_start_x, mean_start_y, mean_target_x, mean_target_y, match_img


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

    result_img_blue = result_sift_blue[4]
    result_img_green = result_sift_green[4]
    result_img_red = result_sift_red[4]

    result_img_blue = result_img_blue[:,:,0]
    result_img_green = result_img_green[:,:,1]
    result_img_red = result_img_red[:,:,2]

    # calculate mean coordinates of matches in target image
    mean_start_x_blue = result_sift_blue[0]
    mean_start_y_blue = result_sift_blue[1]
    mean_start_x_green = result_sift_green[0]
    mean_start_y_green = result_sift_green[1]
    mean_start_x_red = result_sift_red[0]
    mean_start_y_red = result_sift_red[1]
    mean_start_x = int((mean_start_x_blue + mean_start_x_green + mean_start_x_red)/3)
    mean_start_y = int((mean_start_y_blue + mean_start_y_green + mean_start_y_red)/3)

    mean_target_x_blue = result_sift_blue[2]
    mean_target_y_blue = result_sift_blue[3]
    mean_target_x_green = result_sift_green[2]
    mean_target_y_green = result_sift_green[3]
    mean_target_x_red = result_sift_red[2]
    mean_target_y_red = result_sift_red[3]
    mean_target_x = int((mean_target_x_blue + mean_target_x_green + mean_target_x_red)/3)
    mean_target_y = int((mean_target_y_blue + mean_target_y_green + mean_target_y_red)/3)

    result_img = cv2.merge([result_img_red, result_img_green, result_img_blue])

    return mean_start_x, mean_start_y, mean_target_x, mean_target_y, result_img


# calculate position where a particular puzzle piece belongs to
# and move the puzzle piece to this position on the puzzle template
def search_position(piece, template, number_of_pieces_hor, number_of_pieces_ver):

    # crop image of puzzle piece
    piece = crop_image(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))

    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    original_piece_PIL = Image.fromarray(np.asarray(piece))
    template_PIL = Image.fromarray(np.asarray(template))

    # calculate piece size
    piece_width, piece_height = calculate_piece_size(template_PIL, number_of_pieces_hor, number_of_pieces_ver)
    original_piece_width, original_piece_height = original_piece_PIL.size
    
    # calculate position of piece on template
    x_start, y_start, x_target, y_target,_ = improved_sift(piece, template) 

    # resize piece
    resize_width_factor = original_piece_width / piece_width
    resize_height_factor = original_piece_height / piece_height
    resize_factor = (resize_width_factor + resize_height_factor) / 2
    new_piece_width = int(original_piece_width / resize_factor)
    new_piece_height = int(original_piece_height / resize_factor)
    piece = original_piece_PIL.resize((new_piece_width, new_piece_height))

    x_start = int(x_start / resize_width_factor)
    y_start = int(y_start / resize_height_factor)

    # place piece at its position on template
    template_copy = template_PIL.copy()
    template_copy.paste(piece, (x_target-x_start, y_target-y_start))

    return np.asarray(template_copy)