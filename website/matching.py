# imports
import numpy as np
import cv2
from fastdtw import fastdtw
from helper import get_colors, choose_piece, rescale, rotate


def match_tiles(A, B, pieces, screen_pieces):

    LENGTH = 160
    PRECISION = 8
    STEP_A = 20
    STEP_B = 7
    MAX_FORM = 0.015
    MAX_COLOR = 8000
    MAX_PIXEL = 0.03
    MAX_FIT = 0.7
    CENTER = round(LENGTH/2)

    pieceA, pieceB = pieces[A], pieces[B]
    contourA, _ = cv2.findContours(pieceA[:,:,3], 0, 1)
    contourB, _ = cv2.findContours(pieceB[:,:,3], 0, 1)
    contourA, contourB = contourA[0].reshape(-1,2), contourB[0].reshape(-1,2)
    sumLen = contourA.shape[0] + contourB.shape[0]

    # Contour matching
    form_matches = []
    for i in range(0, contourA.shape[0], STEP_A):

      # subcontour A and its type
      subcontourA = np.roll(contourA, -i, 0)[:LENGTH]
      pointA = tuple(np.flip(subcontourA[CENTER]))
      cA, (hA,wA), aA = cv2.minAreaRect(subcontourA)
      typespotA = np.int0(np.flip(subcontourA[0] + subcontourA[-1] - cA))
      typeA = pieceA[:,:,3][tuple(typespotA)]
      a = cv2.drawContours(np.zeros((300,300),'uint8'), subcontourA.reshape(-1,1,2), -1, 255, 1)

      # loop through match subcontours
      for j in range(0, contourB.shape[0], STEP_B):
        
        # subcontour B and its type
        subcontourB = np.roll(contourB, -j, 0)[:LENGTH]
        pointB = tuple(np.flip(subcontourB[CENTER]))
        cB, (hB,wB), aB = cv2.minAreaRect(subcontourB)
        typespotB = np.int0(np.flip(subcontourB[0] + subcontourB[-1] - cB))
        typeB = pieceB[:,:,3][tuple(typespotB)]

        # record good form matches
        if typeB != typeA:
          if ((abs(hA-hB) < PRECISION) & (abs(wA-wB) < PRECISION)) or ((abs(hA-wB) < PRECISION) & (abs(wA-hB) < PRECISION)):
            b = cv2.drawContours(np.zeros((300,300),'uint8'), subcontourB.reshape(-1,1,2), -1, 255, 1)
            fmatch = cv2.matchShapes(a,b,1,0)
            if fmatch < MAX_FORM: 
              colinear = True if np.sign(hA-wA) == np.sign(hB-wB) else False
              if colinear:
                codirect = True if (np.sign(typespotA - np.flip(cA)) ==  np.sign(typespotB - np.flip(cB))).all() else False
              else:
                c = np.concatenate([np.sign(typespotA - np.flip(cA)), np.sign(typespotB - np.flip(cB))])
                codirect = True if (abs(np.sum(c[:3])) + abs(np.sum(c[-3:]))) == 4 else False
              if not colinear: aB = aB + 90
              if not codirect: aB = aB + 180  
              form_matches.append([(i, j), pointA, pointB, round(aB-aA,4), round(fmatch,4)])
  
    # Color matching
    color_matches = []
    for n in range(len(form_matches)):
      (i, j), pointA, pointB, angle, fmatch = form_matches[n]
      subcontourA = np.roll(contourA, -i, 0)[:LENGTH] 
      subcontourB = np.roll(contourB, -j, 0)[:LENGTH]
      colorsA = get_colors(pieceA, subcontourA)
      colorsB = get_colors(pieceB, subcontourB)
      cmatch = fastdtw(colorsA, np.flip(colorsB, axis=0))[0]
      if cmatch < MAX_COLOR: 
        color_matches.append([(i, j), pointA, pointB, angle, fmatch, round(cmatch)])

    # Pre-fitting
    fit_matches = []
    for n in range(len(color_matches)):
      (i, j), pointA, pointB, angle, fmatch, cmatch = color_matches[n]
      a = choose_piece(screen_pieces[A][:,:,3], rescale(pointA, [700,700,0]), 0)
      b = choose_piece(screen_pieces[B][:,:,3], rescale(pointB, [700,700,0]), angle)
      loss = 1 - (np.sum((a+b)>0) / (np.sum(a>0) + np.sum(b>0)))
      contours, _ = cv2.findContours((a+b), 0, 1)
      fit = contours[0].shape[0] / sumLen
      if (loss < MAX_PIXEL) & (fit < MAX_FIT): 
        fit_matches.append([(A, B), (i, j), pointA, pointB, angle, fmatch, cmatch, round(loss+fit,4), 0])

    fit_matches.sort(key=lambda n: n[-1])

    return fit_matches

def get_matches(pieces, screen_pieces):
    # Calculate all possible matches
    matches = []
    for a in range(len(pieces)-1):
        for b in range(a+1,len(pieces)):
            matches.extend(match_tiles(a,b, pieces, screen_pieces))

    # Flip and sort
    for n in range(len(matches)):
        pair, ij, pointa, pointb, angle, fmatch, cmatch, fit, lock = matches[n]
        matches.extend([[(pair[1],pair[0]), ij, pointb, pointa, -angle, fmatch, cmatch, fit, lock]])
    matches.sort(key=lambda m: (m[0], m[-2]))

    return matches