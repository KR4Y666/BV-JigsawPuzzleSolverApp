'''
Praktikum Bildverarbeitung SoSe 2022
Prof. Schilling

Projekt: Jigsaw Puzzle Solver 
Mitglieder: Jens Niebling, Jonas Schmidt, Emma Urban 

Projektbeschreibung: 
Entwerfen eines Programmes, das in der Lage ist, anhand 
vreschiedener Methoden aus der Bildverarbeitung selbständig 
ein Puzzle zu lösen. Dies soll nur anhand eines Bildes der 
Puzzleteile geschehen
'''

'''
Imports
'''

import cv2
import numpy as np
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from fastdtw import fastdtw

'''
Preprocessing of Image
'''


#TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für Abgabe rauslöschen

def showImage(image):
  height = image.shape[0]
  width = image.shape[1]
  height_factor = 15/width*height
  plt.figure(figsize=(15, height_factor))
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.show()

# def showlist(tiles, width=10):
#   n_rows = np.ceil(len(tiles)/5).astype('int')
#   plt.subplots(n_rows, 5, figsize=(width, width))
#   for i in range(len(tiles)):
#     plt.subplot(n_rows, 5, i+1)
#     plt.axis('off')
#     plt.title(str(i))
#     plt.imshow(tiles[i])
#   plt.show()

def preprocess_image(filename):

  # load picture of puzzle pieces using pillow
  img = Image.open(filename)

  # convert picture to RGBA 
  cvt_img = img.convert('RGBA')

  # convert picture to numpy array
  puzzle_img = np.array(cvt_img)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für Abgabe rauslöschen
  #showImage(puzzle_img)

  # convert puzzle picture to Grayscale image
  gray_img = cv2.cvtColor(puzzle_img, cv2.COLOR_RGBA2GRAY)

  # Apply Opencvs adaptive adaptiveThreshold 
  # maybe add finetuning later
  # this separetes puzzle pieces from background
  thresh_img = cv2.adaptiveThreshold(gray_img, 255, 0, 1, 3, 3)

  # add gaussian GaussianBlur
  # maybe adjust Values a little bit later
  blurred_img = cv2.GaussianBlur(thresh_img, (3,3), 1)

  return puzzle_img, blurred_img

# TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
#showImage(blurred_img)

def find_contours(puzzle_img, blurred_img):
  # find contours of pieces using OpenCv
  best_contours, _ = cv2.findContours(blurred_img, 0, 1)

  # sort out number of puzzle pieces -> here max. 25 pieces
  max_pieces = 15
  pieces_sorted = sorted([[cnt.shape[0], i] for i, cnt in enumerate(best_contours)], reverse=True)[:max_pieces]
  valid_pieces = [best_contours[s[1]] for s in pieces_sorted] 

  # fill in contours -> binary Image using Opencv
  pieces_filled = cv2.drawContours(np.zeros(puzzle_img.shape[:2]), valid_pieces, -1, 255, thickness=cv2.FILLED)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  # showImage(pieces_filled)

  # smooth edges using median filter -> adjust size
  edges_smoothed = median_filter(pieces_filled.astype('uint8'), size=10)

  # find better contours
  best_contours, _ = cv2.findContours(edges_smoothed, 0, 1)

  # draw new contours
  cv2.drawContours(edges_smoothed, best_contours, -1, color=0, thickness=1)

  #showImage(edges_smoothed)
  return best_contours

# TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
#showImage(edges_smoothed)

def split_in_pieces(puzzle_img, blurred_img):
  # initialize arrays
  puzzle_pieces = []
  puzzle_piece_centers = []
  pieces_rescaled = []
  best_contours = find_contours(puzzle_img, blurred_img)

  # split picture of all pieces into single ones 
  # TODO change values of image sizes 
  for i in range(len(best_contours)):
    x, y, w, h = cv2.boundingRect(best_contours[i])
    shape, tile = np.zeros(puzzle_img.shape[:2]), np.zeros((300,300,4), 'uint8')
    cv2.drawContours(shape, [best_contours[i]], -1, color=1, thickness=-1)
    shape = (puzzle_img * shape[:,:,None])[y:y+h,x:x+w,:]
    tile[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
    puzzle_pieces.append(tile)
    puzzle_piece_centers.append((h//2+y, w//2+x))

  # change scale of pieces for matching
  #  TODO change canvas size later
  for i in range(len(puzzle_pieces)):
    piece = np.zeros((1400,1400,4), 'uint8')
    piece[550:850, 550:850] = puzzle_pieces[i].copy()
    pieces_rescaled.append(piece)
  
  return puzzle_pieces, pieces_rescaled


'''
Match puzzle pieces
'''
# Helper function to arrage the pieces working with
def choose_pieces(arr_img, point, angle, center=(700,700)):
  image = Image.fromarray(arr_img)
  image = ImageChops.offset(image, center[1] - point[1], center[0] - point[0])
  image = image.rotate(angle)

  return np.array(image)

# Helper function to recale size of puzzle pieces
def rescale(point, position, center=(150,150)):
  cy, cx, angle = position
  if angle!=0: 
    (y, x) = rotate(point, angle, center)
  else: 
    (y, x) = point

  return (y + cy - center[0], x + cx - center[1])

# Helper function to move and rotate piece
def rotate(axis, angle, center=(700,700)):
  dy, dx = center[0]-axis[0], axis[1]-center[1]
  dist = np.sqrt(np.square(axis[0]-center[0]) + np.square(axis[1]-center[1]))
  if dx==0: 
    dx = 1

  base = 90*(1-np.sign(dx)) + np.degrees(np.arctan(dy/dx))
  
  y = round(center[0] - dist * np.sin(np.pi * (base + angle)/180))
  x = round(center[1] + dist * np.cos(np.pi * (base + angle)/180))

  return (y,x)

# Helper function for color matching
def getColors(image, piece_side_contour):
  piece_side_contour = np.flip(piece_side_contour)

  colors = []

  for n in range(len(piece_side_contour)-3):
    (y,x) = piece_side_contour[n]
    (y1,x1) = piece_side_contour[n+3]
    h, w = y1 - y, x1 - x
    colors.append(image[y-w, x+h, :3] + image[y+w, x-h, :3])

  colors = np.array(colors, 'uint8').reshape(-1,1,3)
  colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
  
  return colors.reshape(-1,3)

# matching of puzzle pieces
def match_pieces(input1, input2, puzzle_pieces, pieces_rescaled):

  LENGTH = 160
  PRECISION = 8
  STEP_1 = 20
  STEP_2 = 7
  MAX_FORM = 0.015
  MAX_COLOR = 8000
  MAX_PIXEL = 0.03
  MAX_FIT = 0.77

  CENTER = round(LENGTH/2)

  piece1, piece2 = puzzle_pieces[input1], puzzle_pieces[input2]
  contour1, _ = cv2.findContours(piece1[:,:,3], 0, 1)
  contour2, _ = cv2.findContours(piece2[:,:,3], 0, 1)
  contour1, contour2 = contour1[0].reshape(-1,2), contour2[0].reshape(-1,2)
  sumLen = contour1.shape[0] + contour2.shape[0]

  # Contour matching
  contour_matches = []
  for i in range(0, contour1.shape[0], STEP_1):

    # piece side contour of piece1 and type of contours
    side1 = np.roll(contour1, -i, 0)[:LENGTH]
    point1 = tuple(np.flip(side1[CENTER]))
    c1, (h1,w1), a1 = cv2.minAreaRect(side1)
    typepoint1 = np.int0(np.flip(side1[0] + side1[-1] - c1))
    type1 = piece1[:,:,3][tuple(typepoint1)]
    a = cv2.drawContours(np.zeros((300,300),'uint8'), side1.reshape(-1,1,2), -1, 255, 1)

    for j in range(0, contour2.shape[0], STEP_2):
      
      # piece side contour of piece2 and its type
      side2 = np.roll(contour2, -j, 0)[:LENGTH]
      point2 = tuple(np.flip(side2[CENTER]))
      c2, (h2,w2), a2 = cv2.minAreaRect(side2)
      typepoint2 = np.int0(np.flip(side2[0] + side2[-1] - c2))
      type2 = piece2[:,:,3][tuple(typepoint2)]

      # compute best matches via precision
      if type2 != type1:
        if ((abs(h1-h2) < PRECISION) & (abs(w1-w2) < PRECISION)) or ((abs(h1-w2) < PRECISION) & (abs(w1-h2) < PRECISION)):
          b = cv2.drawContours(np.zeros((300,300),'uint8'), side2.reshape(-1,1,2), -1, 255, 1)
          fmatch = cv2.matchShapes(a,b,1,0)
          if fmatch < MAX_FORM: 
            colinear = True if np.sign(h1-w1) == np.sign(h2-w2) else False
            if colinear:
              codirect = True if (np.sign(typepoint1 - np.flip(c1)) ==  np.sign(typepoint2 - np.flip(c2))).all() else False
            else:
              c = np.concatenate([np.sign(typepoint1 - np.flip(c1)), np.sign(typepoint2 - np.flip(c2))])
              codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
            if not colinear: a2 = a2 + 90
            if not codirect: a2 = a2 + 180  
            contour_matches.append([(i, j), point1, point2, round(a2-a1,4), round(fmatch,4)])
 
  # color matching along piece edges
  color_matches = []
  for n in range(len(contour_matches)):
    (i, j), point1, point2, angle, fmatch = contour_matches[n]
    side1 = np.roll(contour1, -i, 0)[:LENGTH] 
    side2 = np.roll(contour2, -j, 0)[:LENGTH]
    colors1 = getColors(piece1, side1)
    colors2 = getColors(piece2, side2)
    cmatch = fastdtw(colors1, np.flip(colors2, axis=0))[0]
    if cmatch < MAX_COLOR: 
      color_matches.append([(i, j), point1, point2, angle, fmatch, round(cmatch)])

  # pre fitting of puzzle pieces
  fit_matches = []
  for n in range(len(color_matches)):
    (i, j), point1, point2, angle, fmatch, cmatch = color_matches[n]
    a = choose_pieces(pieces_rescaled[input1][:,:,3], rescale(point1, [700,700,0]), 0)
    b = choose_pieces(pieces_rescaled[input2][:,:,3], rescale(point2, [700,700,0]), angle)
    loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
    contours, _ = cv2.findContours((a+b), 0, 1)
    fit = contours[0].shape[0] / sumLen

    if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
      fit_matches.append([(input1, input2), (i, j), point1, point2, angle, fmatch, cmatch, round(loss+fit,4), 0])

  fit_matches.sort(key=lambda n: n[-1])

  return fit_matches

def match(puzzle_pieces, pieces_rescaled):
  # compute possible matches of all pieces
  all_matches = []
  for a in range(len(puzzle_pieces)-1):
    for b in range(a+1,len(puzzle_pieces)):
      all_matches.extend(match_pieces(a,b, puzzle_pieces, pieces_rescaled))

  # Flip and sort
  for n in range(len(all_matches)):
    pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = all_matches[n]
    all_matches.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
  all_matches.sort(key=lambda m: (m[0], m[-2]))

  return all_matches


'''
Assemble matched pieces
'''

def unpuzzle(canvas, positions, input1, input2, point1, point2, angle1, angle2):
  
  # push pieces on "working surface" aka canvas
  for N, pos in enumerate(positions):
    if N in canvas:
      new_center = (pos[0] + 700 - point1[0], pos[1] + 700 - point1[1])
      new_center = rotate(new_center, angle1)
      new_angle = pos[2] + angle1
      positions[N] = [*new_center, new_angle]

  canvas.append(input2)
  center = rotate((700 + 700 - point2[0], 700 + 700 - point2[1]), angle2)
  positions[input2] = [*center, angle2]

  return canvas, positions

# unpuzzle pieces -> put together
def unpuzzle_pieces(pieces_rescaled, puzzle_pieces, all_matches):
  assembly = pieces_rescaled[0].copy()
  positions = [[0,0,0]]*len(puzzle_pieces)
  positions[0] = [700,700,0]
  canvas = [0]
  attempts = 0

  while (len(canvas) < 15) & (attempts < 10):
    for n in range(len(all_matches)):
      
      (input1, input2), ij, point1, point2, angle2, _, _, _, lock = all_matches[n]
      point1 = rescale(point1, positions[input1])
      point2 = rescale(point2, (700,700,0))

      if input1 in canvas:
        angle1 = - positions[input1][2]
        pre_assembly = choose_pieces(assembly.copy(), point1, angle1)
        
        if input2 not in canvas:
          newtile = choose_pieces(pieces_rescaled[input2], point2, angle2)

          # !!!plagiat fix or pass depending on loss of pixels
          loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(newtile[:,:,3]>0) - 
                  np.sum((pre_assembly+newtile)[:,:,3]>0)
                  ) / np.sum(newtile[:,:,3]>0)
          if loss < 0.1: 
            all_matches[n][-1] = 1
            assembly = pre_assembly.copy() + newtile.copy()
            canvas, positions = unpuzzle(canvas, positions, 
                                            point1, point2, angle1, angle2)
    
    attempts += 1

    return assembly

def solve_puzzle(filename):
  puzzle_img, blurred_img = preprocess_image(filename)
  best_contours = find_contours(puzzle_img, blurred_img)
  puzzle_pieces, pieces_rescaled = split_in_pieces(puzzle_img, blurred_img)
  all_matches = match(puzzle_pieces, pieces_rescaled)
  assembly = unpuzzle_pieces(pieces_rescaled, puzzle_pieces, all_matches)

  showImage(assembly)
  
solve_puzzle('puzzle.png')
#showImage(assembly)