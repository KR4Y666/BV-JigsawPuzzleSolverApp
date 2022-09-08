
from scipy.ndimage import filters
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import filters


def preprocessing(puzzle_filename, pieces_number):

  #loading picture of puzlle pieces, converting them to RGBA and finaly converting them to numpy array
  puzzle = np.array(Image.open(puzzle_filename).convert('RGBA'))
  
  #use of function - details below
  sill = adaptive_thresholding(puzzle)

  #use of function - details below
  contours, tender = find_and_fill_contours(puzzle, sill, pieces_number)

  #use of function - details below
  pieces, piece_centers, screen_pieces = split_into_pieces(puzzle, contours, tender)
  
  return puzzle, pieces, piece_centers, screen_pieces


def adaptive_thresholding(puzzle):

    #convert puzzle picture to grayscale image
    sill = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
    #apply opencv adaptivethreshold to separet puzzle pieces from background
    sill = cv2.adaptiveThreshold(sill, 255, 0, 1, 3, 3)
    #apply gaussianblur to image
    sill = cv2.GaussianBlur(sill, (3,3), 1)

    return sill

def find_and_fill_contours(puzzle, sill, pieces_number):
    #apply opencvs findcountours function
    contours, _ = cv2.findContours(sill, 0, 1)
    #sort out the number of puzzle pieces , maximum gets generated dynamically 
    sorting = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:pieces_number]
    max = [contours[s[1]] for s in sorting] 
    #using opencv binary image we fill in contours
    fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), max, -1, 255, thickness=cv2.FILLED)

    #tender edges using median filter and adjust size
    tender = filters.median_filter(fill.astype('uint8'), size=1)
    #opencv funtion to find better contours
    trim_contours, _ = cv2.findContours(tender, 0, 1)
    #opencv funtion to draw new contours
    cv2.drawContours(tender, trim_contours, -1, color=0, thickness=1)

    return contours, tender

def split_into_pieces(puzzle, contours, tender):

    #find better contours
    contours, _ = cv2.findContours(tender, 0, 1)
    #initialize arrays
    pieces = []
    piece_centers = []

    #split picture of all puzzle pieces into a single ones
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        shape, piece = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
        cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
        shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
        piece[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
        pieces.append(piece)
        piece_centers.append((h//2+y, w//2+x))
    
    #initialize array
    screen_pieces = []
    #change scale of pieces for better matching results
    for i in range(len(pieces)):
        screen_piece = np.zeros((1400,1400,4), 'uint8')
        screen_piece[550:850, 550:850] = pieces[i].copy()
        screen_pieces.append(screen_piece)

    return pieces, piece_centers, screen_pieces