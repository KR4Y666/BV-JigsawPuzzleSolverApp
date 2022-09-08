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
import math 
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.ndimage import filters
from scipy.ndimage import median_filter
from fastdtw import fastdtw

'''
Preprocessing of Image
'''


#TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen

def showImage(image, width=15):
  plt.figure(figsize=(width, width/1000*727))
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.show()


def solve_puzzle(filename):

  # load in Picture of puzzle pieces using pillow
  picture = Image.open(filename)

  # convert picture to RGBA 
  converted_picture = picture.convert('RGBA')

  # convert picture to numpy array
  picture_as_array = np.array(converted_picture)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  # showpic(picture_as_array) 

  # convert puzzle picture to Grayscale image
  picture_grayscale = cv2.cvtColor(picture_as_array, cv2.COLOR_RGBA2GRAY)

  # Apply Opencvs adaptive adaptiveThreshold 
  # maybe add finetuning later
  # this separetes puzzle pieces from background
  adaptive_threshold = cv2.adaptiveThreshold(picture_grayscale, 255, 0, 1, 3, 3)

  # add gaussian GaussianBlur
  # maybe adjust Values a little bit later
  picture_blurred = cv2.GaussianBlur(adaptive_threshold, (3,3), 1)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  #showpic(picture_blurred)

  # find contours of pieces using Opencvs
  contours, _ = cv2.findContours(picture_blurred, 0, 1)

  # sort out number of puzzle pieces -> here max puzzle size 25 pieces
  pieces_sorted = sorted([[cnt.shape[0], i] for i, cnt in enumerate(contours)], reverse=True)[:10]
  true_pieces = [contours[p[1]] for p in pieces_sorted] 

  # fill in contours -> binary Image using Opencv
  pieces_filled = cv2.drawContours(np.zeros(picture_as_array.shape[:2]), true_pieces, -1, 255, thickness=cv2.FILLED)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  #showImage(pieces_filled)

  # smooth edges using median filter -> adjust size
  edges_smoothed = median_filter(pieces_filled.astype('uint8'), size=12)

  # cut out errors -> better contours
  best_contours, _ = cv2.findContours(edges_smoothed, 0, 1)

  # draw new contours
  cv2.drawContours(edges_smoothed, best_contours, -1, color=0, thickness=1)

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  showImage(edges_smoothed)

  # find better contours
  contours, _ = cv2.findContours(edges_smoothed, 0, 1)

  # init arrays
  puzzle_pieces = []
  puzzle_pieces_center = []

  # split picture of all pieces into single ones 
  # TODO change values of image sizes 
  for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    shape, piece = np.zeros(picture_as_array.shape[:2]), np.zeros((300,300,4), 'uint8')
    cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
    shape = (picture_as_array * shape[:,:,None])[y:y+h,x:x+w,:]
    piece[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
    puzzle_pieces.append(piece)
    puzzle_pieces_center.append((h//2+y, w//2+x))

  '''
  Match puzzle pieces
  '''
  print("here1")
  # init array 
  pieces_rescaled = []

  # chanege scale of pieces for matching
  #  TODO change canvas size later
  for i in range(len(puzzle_pieces)):
    working_piece = np.zeros((1400,1400,4), 'uint8')
    working_piece[550:850, 550:850] = puzzle_pieces[i].copy()
    pieces_rescaled.append(pieces_rescaled)

  #Helperfunction
  # recale size of puzzle pieces
  def rescale(axis, position, center=(150,150)):
    cy, cx, angle = position
    
    if angle!=0:
        (y, x) = rotate(axis, angle, center)
    else: 
        (y, x) = axis

    return (y + cy - center[0], x + cx - center[1])

  # Helperfunction
  # function to rotate piece
  def rotate(axis, angle, center=(700,700)):
    dy, dx = center[0]-axis[0], axis[1]-center[1]
    dist = np.sqrt(np.square(axis[0]-center[0]) + np.square(axis[1]-center[1]))
    
    if dx==0:
        dx = 1
        
    base = 90*(1-np.sign(dx)) + np.degrees(np.arctan(dy/dx))
    
    y = round(center[0] - dist * np.sin(np.pi * (base + angle)/180))
    x = round(center[1] + dist * np.cos(np.pi * (base + angle)/180))

    return (y,x)

  print("here2")
  # Helperfunction
  # arrage pieces to work with
  def choose_pieces(arr_img, point, angle, center=(700,700)):
    image = Image.fromarray(arr_img)
    image = ImageChops.offset(image, center[1] - point[1], center[0] - point[0])
    image = image.rotate(angle)

    return np.array(image)
  print("here3")
  # Helper function for color matching
  def compute_edge_color(pict, piece_side_contour):
      piece_side_contour = np.flip(piece_side_contour)

      colors = []
      
      for n in range(len(piece_side_contour)-3):
        (y,x) = piece_side_contour[n]
        (y1,x1) = piece_side_contour[n+3]
        h, w = y1 - y, x1 - x
        colors.append(pict[y-w, x+h, :3] + pict[y+w, x-h, :3])

      colors = np.array(colors, 'uint8').reshape(-1,1,3)
      colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
    
      return colors.reshape(-1,3)
  print("here4")
  # matching of puzzle pices
  def matching(input1, input2):

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
    center1, _ = cv2.findContours(piece1[:,:,3], 0, 1)
    center2, _ = cv2.findContours(piece2[:,:,3], 0, 1)
    center1, center2 = center1[0].reshape(-1,2), center2[0].reshape(-1,2)
    length_sum = center1.shape[0] + center2.shape[0]

    form_matches = []
    for i in range(0, center1.shape[0], STEP_1):

      # piece_side_contour of piece1 and type of contours
      side1 = np.roll(center1, -i, 0)[:LENGTH]
      point1 = tuple(np.flip(side1[CENTER]))
      c1, (h1,w1), a1 = cv2.minAreaRect(side1)
      typepoint1 = np.int0(np.flip(side1[0] + side1[-1] - c1))
      type1 = piece1[:,:,3][tuple(typepoint1)]
      a = cv2.drawContours(np.zeros((300,300),'uint8'), side1.reshape(-1,1,2), -1, 255, 1)

      for j in range(0, center2.shape[0], STEP_2):
        
        # piece_side_contour of piece2 and its type
        side2 = np.roll(center2, -j, 0)[:LENGTH]
        point2 = tuple(np.flip(side2[CENTER]))
        c2, (h2,w2), a2 = cv2.minAreaRect(side2)
        typepoint2 = np.int0(np.flip(side2[0] + side2[-1] - c2))
        type2 = piece2[:,:,3][tuple(typepoint2)]

        # compute  best matches via Precission
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
              form_matches.append([(i, j), point1, point2, round(a2-a1,4), round(fmatch,4)])
              
    # init array for colormatching
    color_matches = []
    print("here6")
    # color matching along edges  
    for n in range(len(form_matches)):
        (i, j), point1, point2, angle, fmatch = form_matches[n]
        side1 = np.roll(center1, -i, 0)[:LENGTH] 
        side1B = np.roll(center2, -j, 0)[:LENGTH]
        colors1 = compute_edge_color(piece1, side1)
        colors2 = compute_edge_color(piece2, side2)
        cmatch = fastdtw(colors1, np.flip(colors2, axis=0))[0]
        if cmatch < MAX_COLOR: 
          color_matches.append([(i, j), point1, point2, angle, fmatch, round(cmatch)])

  # init array for fitting matches 
    fit_matches = []
    print("here10")

    # pre fitting of puzzle pieces
    for n in range(len(color_matches)):
      (i, j), point1, point2, angle, fmatch, cmatch = color_matches[n]
      a = choose_pieces(pieces_rescaled[input1][:,:,3], rescale(point1, [700,700,0]), 0)
      b = choose_pieces(pieces_rescaled[input2][:,:,3], rescale(point2, [700,700,0]), angle)
      loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
      contours, _ = cv2.findContours((a+b), 0, 1)
      fit = contours[0].shape[0] / length_sum
      
      if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
          fit_matches.append([(input1, input2), (i, j), point1, point2, angle, fmatch, cmatch, round(loss+fit,4), 0])

      fit_matches.sort(key=lambda n: n[-1])

      return fit_matches

  # init array for possible matches
  matches_all = []

  # commpute all possible matches of pieces
  for a in range(len(puzzle_pieces)-1):
    for b in range(i+1,len(puzzle_pieces)):
      matches_all.extend(matching(a,b))

  # sort matches and flip in same step
  for n in range(len(matches_all)):
    pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = matches_all[n]
    matches_all.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
  matches_all.sort(key=lambda m: (m[0], m[-2]))

  '''
  Assemble matched pieces
  '''

  def unpuzzle(canvas, positions, point1, point2, angle1, angle2):
    
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

#unpuzzle pieces -> put together
  assembly = pieces_rescaled[0].copy()
  positions = [[0,0,0]]*len(puzzle_pieces)
  positions[0] = [700,700,0]
  canvas = [0]
  attempts = 0

  while (len(canvas) < 15) & (attempts < 10):
    for n in range(len(matches_all)):
          
      (input1, input2), ij, point1, point2, angle2, _, _, _, lock = matches_all[n]
      point1 = rescale(point1, positions[input1])
      point2 = rescale(point2, (700,700,0))

      if input1 in canvas:
        angle1 = - positions[input1][2]
        pre_assembly = choose_pieces(assembly.copy(), point1, angle1)
        
        if input2 not in canvas:
          new_piece = choose_pieces(pieces_rescaled[input2], point2, angle2)

          # fix or pass depending on loss of pixels
          loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(new_piece[:,:,3]>0) - 
                  np.sum((pre_assembly+new_piece)[:,:,3]>0)
                  ) / np.sum(new_piece[:,:,3]>0)
          if loss < 0.1: 
            matches_all[n][-1] = 1
            assembly = pre_assembly.copy() + new_piece.copy()
            canvas, positions = unpuzzle(canvas, positions, 
                                            point1, point2, angle1, angle2)
    
    attempts += 1

  # TODO Nur für zwischenergebnisse für Präsentation und Ausarbeitung. Für ABgabe rauslöschen
  showImage(assembly)
  return assembly

solve_puzzle('puzzle.png')

