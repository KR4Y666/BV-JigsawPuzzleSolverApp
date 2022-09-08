# imports
import numpy as np
import cv2
from helper import choose_piece, rescale, rotate

def updateCanvas(screen, locations, pointA, pointB, cornerA, cornerB, B):
    
    # update records for tiles on canvas
    for N, pos in enumerate(locations):
      if N in screen:
        new_center = (pos[0] + 700 - pointA[0], pos[1] + 700 - pointA[1])
        new_center = rotate(new_center, cornerA)
        new_corner = pos[2] + cornerA
        locations[N] = [*new_center, new_corner]

    # append record for the added tile
    screen.append(B)
    center = rotate((700 + 700 - pointB[0], 700 + 700 - pointB[1]), cornerB)
    locations[B] = [*center, cornerB]

    return screen, locations


def assemble(puzzle, pieces, screen_pieces, matches, piece_centers):
  # Assembly
  assembly = screen_pieces[0].copy()
  locations = [[0,0,0]]*len(pieces)
  locations[0] = [700,700,0]
  screen = [0]
  attempts = 0

  while (len(screen) < 15) & (attempts < 10):
    for n in range(len(matches)):
          
      # take next matching pair
      (A, B), ij, pointA, pointB, angleB, _, _, _, lock = matches[n]
      pointA = rescale(pointA, locations[A])
      pointB = rescale(pointB, (700,700,0))

      if A in screen:
        cornerA = - locations[A][2]
        pre_assembly = choose_piece(assembly.copy(), pointA, cornerA)
        
        if B not in screen:
          new_piece = choose_piece(screen_pieces[B], pointB, angleB)

          # fix or pass depending on loss of pixels
          loss = (np.sum(pre_assembly[:,:,3]>0) + np.sum(new_piece[:,:,3]>0) - 
                  np.sum((pre_assembly+new_piece)[:,:,3]>0)
                  ) / np.sum(new_piece[:,:,3]>0)
          if loss < 0.1: 
            matches[n][-1] = 1
            assembly = pre_assembly.copy() + new_piece.copy()
            screen, locations = updateCanvas(screen, locations, 
                                            pointA, pointB, cornerA, angleB, B)
    
    attempts += 1

  # Mark matches in original image
  count = 0
  markup = puzzle.copy()
  colors = [[r,g,b,255] for r in [255,100,0] for g in [255,100,0] for b in [255,100,0]]
  for n in range(len(matches)):
    (A, B), _, pointA, pointB, _, _, _, _, lock = matches[n]
    if lock == 1:
      count += 1
      centerA = (piece_centers[A][1]-(150-pointA[1]), piece_centers[A][0]-(150-pointA[0]))
      centerB = (piece_centers[B][1]-(150-pointB[1]), piece_centers[B][0]-(150-pointB[0]))
      cv2.circle(markup, centerA, 15, colors[count], -1)
      cv2.circle(markup, centerB, 15, colors[count], -1)
      cv2.putText(markup, str(count), (centerA[0]-7,centerA[1]+5), 1, 1, [255,255,255,255], 2)
      cv2.putText(markup, str(count), (centerB[0]-7,centerB[1]+5), 1, 1, [255,255,255,255], 2)

  return assembly