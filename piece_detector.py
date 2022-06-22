from __future__ import print_function
import sys
import cv2 as cv
from piece_segmentation_via_contours import crop_image
from PIL import Image

use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5


def main():

    global img
    global templ
    img = cv.imread("puzzle_data\Puzzle_Template_Feuerwehr.jpg", cv.IMREAD_COLOR)
    templ = cv.imread("puzzle_data\Puzzle_Piece_Feuerwehr_cut.jpg", cv.IMREAD_COLOR)
    templ = crop_image(templ)
    templ = cv.resize(templ, (200,200))
    
    cv.namedWindow( image_window, cv.WINDOW_AUTOSIZE )
    cv.namedWindow( result_window, cv.WINDOW_AUTOSIZE )
    
    
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    cv.createTrackbar( trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod )
    
    MatchingMethod(match_method)
    
    cv.waitKey(0)
    return 0
    
def MatchingMethod(param):
    global match_method
    match_method = param
    
    img_display = img.copy()
    
    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        result = cv.matchTemplate(img, templ, match_method, None, mask)
    else:
        result = cv.matchTemplate(img, templ, match_method)
    
    
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    
    
    if (match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED):
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    
    #scale output image
    scale_percent = 20
    width = int(img_display.shape[1] * scale_percent / 100)
    height = int(img_display.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0,0,0), 2, 8, 0 )
    cv.imshow(image_window, cv.resize(img_display, dim))
    cv.imshow(result_window, result)
    
    pass

main()