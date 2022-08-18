import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from  PIL  import Image

#remove background of an image
def remove_background_of(img):

    original = img.copy()

    l = int(max(5, 6))
    u = int(min(6, 6))

    edges = cv.GaussianBlur(img, (21, 51), 3)
    edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(edges, l, u)

    _, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY  + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

    data = mask.tolist()
    sys.setrecursionlimit(10**8)
    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0

    mask = np.array(image, np.uint8)

    result = cv.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255
    plt.imshow(result)
    plt.show()

    return result

img = cv.imread('./puzzle_data/puzzle4_1.jpg')
res = remove_background_of(img)
