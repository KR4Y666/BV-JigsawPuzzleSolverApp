###################################################################
"""
Projekt für Kurs "Praktikum Bildverarbeitung" an der Universität Tübingen
Betreuender Professor: Prof. Schilling 

Gruppenmitglieder: 
Jonas Schmidt
Emma Urban
Jens Niebling

Ziel des Projekts ist ein Programm, das in der Lage ist, anhand von 
Methoden der Bildverarbeitung ein ungelöstes Puzzle anhand von Bildern 
selbständig zu lösen
"""
###################################################################


###################################################################
# Imports 
###################################################################
import math 
import os 
import sys 
from scipy.ndimage import filters
from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from fastdtw import fastdtw
import numpy as np
import cv2
###################################################################


###################################################################
# Preprocessing
###################################################################
# Function to plot picture of results of the different steps
# comment plotResult out, if plot of step not necessary or wanted
# Maybe define sizes: plotResults(image, width = 10, height = 10)
# plt.figure(figsize = (width, height))
def plotResults(image):
  plt.figure()
  plt.imshow(image, cmap='gray')
  plt.axis('off')
  plt.show()

# Load image of puzzle pieces 
# convert picture to RGBA and to Numpay Array
# Using Pillow for opening and converting 
puzzleImage = Image.open('puzzle4.png')
puzzleImage = puzzleImage.convert('RGBA')
puzzle = np.array(puzzleImage)
# plotResults(puzzle)

# convert picture of puzzle into grayscale image
puzzleGray = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)

# Apply opencvs adatipve thresholding technique
# maybe add some finetuning here
threshold = cv2.adaptiveThreshold(puzzleGray, 255, 0, 1, 3, 3)

# blur picuture a little bit using gaussian blur 
threshold = cv2.GaussianBlur(threshold, (3,3), 2)
#plot_results(thresh)

# compute contours of image using opencv findContours
contours, _ = cv2.findContours(threshold, 0, 1)

# draw and plot contours 
fill = cv2.drawContours(np.zeros(puzzle.shape[:2]), contours, -1, 255, thickness=cv2.FILLED)
#plot_results(fill)

# smooth contours using median filter from scipy package
# for different smoothings, adjust size = x 
# changing size necessary for photos or screenshots ... 
smooth = filters.median_filter(fill.astype('uint8'), size=12)

# compute new filtered contours and draw them
trim_contours, _ = cv2.findContours(smooth, 0, 1)
cv2.drawContours(smooth, trim_contours, -1, color=0, thickness=1)
#plot_results(smooth)

# compute single puzzle pieces from contours image 
contours, _ = cv2.findContours(smooth, 0, 1)
pieces, pieceCenters = [], []
for i in range(len(contours)):
  x, y, w, h = cv2.boundingRect(contours[i])
  shape, piece = np.zeros(puzzle.shape[:2]), np.zeros((300,300,4), 'uint8')
  cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)
  shape = (puzzle * shape[:,:,None])[y:y+h,x:x+w,:]
  piece[(300-h)//2:(300-h)//2+h,(300-w)//2:(300-w)//2+w] = shape
  pieces.append(piece)
  pieceCenters.append((h//2+y, w//2+x))

'''
# show puzzle pieces in Figure 
n_rows = np.ceil(len(tiles)/5).astype('int')       
plt.subplots(n_rows, 5, figsize=(10, 10))
for i in range(len(tiles)):
  plt.subplot(n_rows, 5, i+1)
  plt.axis('off')
  plt.title(str(i))
  plt.imshow(tiles[i])
plt.show()
'''
###################################################################


###################################################################
# Matching of Puzzle Pieces
#################################################################
# Rescale pieces to assembly format
canvasPieces = []
for i in range(len(pieces)):
  canvasPiece = np.zeros((1400,1400,4), 'uint8')
  canvasPiece[550:850, 550:850] = pieces[i].copy()
  canvasPieces.append(canvasPiece)

# ???
def getColors(image, subcontour):
  subcontour = np.flip(subcontour)

  colors = []
  for n in range(len(subcontour)-3):
    (y,x) = subcontour[n]
    (y1,x1) = subcontour[n+3]
    h, w = y1 - y, x1 - x
    colors.append(image[y-w, x+h, :3] + image[y+w, x-h, :3])

  colors = np.array(colors, 'uint8').reshape(-1,1,3)
  colors = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV)
  
  return colors.reshape(-1,3)

# move and rotate puzzle piece
def positionPiece(arr_img, point, angle, center=(700,700)):
  img = Image.fromarray(arr_img)
  img = ImageChops.offset(img, center[1] - point[1], center[0] - point[0])
  img = img.rotate(angle)

  return np.array(img)

# calculate the rotation point (import for moving and rotating puzzle piece)
def rotatePoint(point, angle, center=(700,700)):
  dy, dx = center[0]-point[0], point[1]-center[1]
  distance = np.sqrt(np.square(point[0]-center[0]) + np.square(point[1]-center[1]))
  if dx==0: dx = 1
  base = 90*(1-np.sign(dx)) + np.degrees(np.arctan(dy/dx))
  
  y = round(center[0] - distance * np.sin(np.pi * (base + angle)/180))
  x = round(center[1] + distance * np.cos(np.pi * (base + angle)/180))

  return (y,x)

# translate coordinates of puzzle piece image to coordinates of puzzle canvas
def rescale(point, position, center=(150,150)):
  cy, cx, angle = position
  if angle!=0: (y, x) = rotatePoint(point, angle, center)
  else: (y, x) = point

  return (y + cy - center[0], x + cx - center[1])


def matchPieces(piece1, piece2):

  length = 160
  precision = 7
  STEP_A = 20
  STEP_B = 7
  MAX_FORM = 0.015
  MAX_COLOR = 8000
  MAX_PIXEL = 0.03
  MAX_FIT = 0.77

  CENTER = round(length/2)

  pieceA, pieceB = pieces[piece1], pieces[piece2]
  contourA, _ = cv2.findContours(pieceA[:,:,3], 0, 1)
  contourB, _ = cv2.findContours(pieceB[:,:,3], 0, 1)
  contourA, contourB = contourA[0].reshape(-1,2), contourB[0].reshape(-1,2)
  sumLen = contourA.shape[0] + contourB.shape[0]

  # Contour matching
  contourMatches = []
  for i in range(0, contourA.shape[0], STEP_A):

    # subcontour A and its type
    subcA = np.roll(contourA, -i, 0)[:length]
    pointA = tuple(np.flip(subcA[CENTER]))
    cA, (hA,wA), aA = cv2.minAreaRect(subcA)
    typepointA = np.int0(np.flip(subcA[0] + subcA[-1] - cA))
    typeA = pieceA[:,:,3][tuple(typepointA)]
    a = cv2.drawContours(np.zeros((300,300),'uint8'), subcA.reshape(-1,1,2), -1, 255, 1)

    # loop through match subcontours
    for j in range(0, contourB.shape[0], STEP_B):
      
      # subcontour B and its type
      subcB = np.roll(contourB, -j, 0)[:length]
      pointB = tuple(np.flip(subcB[CENTER]))
      cB, (hB,wB), aB = cv2.minAreaRect(subcB)
      typepointB = np.int0(np.flip(subcB[0] + subcB[-1] - cB))
      typeB = pieceB[:,:,3][tuple(typepointB)]

      # record good form matches
      if typeB != typeA:
        if ((abs(hA-hB) < precision) & (abs(wA-wB) < precision)) or ((abs(hA-wB) < precision) & (abs(wA-hB) < precision)):
          b = cv2.drawContours(np.zeros((300,300),'uint8'), subcB.reshape(-1,1,2), -1, 255, 1)
          fmatch = cv2.matchShapes(a,b,1,0)
          if fmatch < MAX_FORM: 
            colinear = True if np.sign(hA-wA) == np.sign(hB-wB) else False
            if colinear:
              codirect = True if (np.sign(typepointA - np.flip(cA)) ==  np.sign(typepointB - np.flip(cB))).all() else False
            else:
              c = np.concatenate([np.sign(typepointA - np.flip(cA)), np.sign(typepointB - np.flip(cB))])
              codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
            if not colinear: aB = aB + 90
            if not codirect: aB = aB + 180  
            contourMatches.append([(i, j), pointA, pointB, round(aB-aA,4), round(fmatch,4)])
 
  # Color matching
  colorMatches = []
  for n in range(len(contourMatches)):
    (i, j), pointA, pointB, angle, fmatch = contourMatches[n]
    subcA = np.roll(contourA, -i, 0)[:length] 
    subcB = np.roll(contourB, -j, 0)[:length]
    colorsA = getColors(pieceA, subcA)
    colorsB = getColors(pieceB, subcB)
    cmatch = fastdtw(colorsA, np.flip(colorsB, axis=0))[0]
    if cmatch < MAX_COLOR: 
      colorMatches.append([(i, j), pointA, pointB, angle, fmatch, round(cmatch)])

  # Pre-fitting
  fitMatches = []
  for n in range(len(colorMatches)):
    (i, j), pointA, pointB, angle, fmatch, cmatch = colorMatches[n]
    a = positionPiece(canvasPieces[piece1][:,:,3], rescale(pointA, [700,700,0]), 0)
    b = positionPiece(canvasPieces[piece2][:,:,3], rescale(pointB, [700,700,0]), angle)
    loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
    contours, _ = cv2.findContours((a+b), 0, 1)
    fit = contours[0].shape[0] / sumLen
    if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
      fitMatches.append([(piece1, piece2), (i, j), pointA, pointB, angle, fmatch, cmatch, round(loss+fit,4), 0])

  fitMatches.sort(key=lambda n: n[-1])

  return fitMatches

# Calculate all possible matches
matches = []
for a in range(len(pieces)-1):
  for b in range(a+1,len(pieces)):
    matches.extend(matchPieces(a,b))

# Flip and sort
for n in range(len(matches)):
  pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = matches[n]
  matches.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
matches.sort(key=lambda m: (m[0], m[-2]))
###################################################################

###################################################################
# Moving Matches 
###################################################################


def updateCanvas(canvas, positions, pointA, pointB, angleA, angleB):
  
  # update records for tiles on canvas
  for N, pos in enumerate(positions):
    if N in canvas:
      new_center = (pos[0] + 700 - pointA[0], pos[1] + 700 - pointA[1])
      new_center = rotatePoint(new_center, angleA)
      new_angle = pos[2] + angleA
      positions[N] = [*new_center, new_angle]

  # append record for the added tile
  canvas.append(B)
  center = rotatePoint((700 + 700 - pointB[0], 700 + 700 - pointB[1]), angleB)
  positions[B] = [*center, angleB]

  return canvas, positions

# Assembly
assembly = canvasPieces[0].copy()
positions = [[0,0,0]]*len(pieces)
positions[0] = [700,700,0]
canvas = [0]
attempts = 0

while (len(canvas) < 15) & (attempts < 10):
  for n in range(len(matches)):
        
    # take next matching pair
    (A, B), ij, pointA, pointB, angleB, _, _, _, lock = matches[n]
    pointA = rescale(pointA, positions[A])
    pointB = rescale(pointB, (700,700,0))

    if A in canvas:
      angleA = - positions[A][2]
      preAssembly = positionPiece(assembly.copy(), pointA, angleA)
      
      if B not in canvas:
        newtile = positionPiece(canvasPieces[B], pointB, angleB)

        # fix or pass depending on loss of pixels
        loss = (np.sum(preAssembly[:,:,3]>0) + np.sum(newtile[:,:,3]>0) - 
                np.sum((preAssembly+newtile)[:,:,3]>0)
                ) / np.sum(newtile[:,:,3]>0)
        if loss < 0.1: 
          matches[n][-1] = 1
          assembly = preAssembly.copy() + newtile.copy()
          canvas, positions = updateCanvas(canvas, positions, 
                                           pointA, pointB, angleA, angleB)
  
  attempts += 1

plotResults(assembly)

# Mark matches in original image
'''
count = 0
markup = puzzle.copy()
colors = [[r,g,b,255] for r in [255,100,0] for g in [255,100,0] for b in [255,100,0]]
for n in range(len(matches)):
  (A, B), _, pointA, pointB, _, _, _, _, lock = matches[n]
  if lock == 1:
    count += 1
    centerA = (tile_centers[A][1]-(150-pointA[1]), tile_centers[A][0]-(150-pointA[0]))
    centerB = (tile_centers[B][1]-(150-pointB[1]), tile_centers[B][0]-(150-pointB[0]))
    cv2.circle(markup, centerA, 15, colors[count], -1)
    cv2.circle(markup, centerB, 15, colors[count], -1)
    cv2.putText(markup, str(count), (centerA[0]-7,centerA[1]+5), 1, 1, [255,255,255,255], 2)
    cv2.putText(markup, str(count), (centerB[0]-7,centerB[1]+5), 1, 1, [255,255,255,255], 2)

plot_results(markup)
'''
###################################################################
