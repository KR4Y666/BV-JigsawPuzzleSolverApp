#! C:\Python\Python310\python.exe
import cgi, os
import cgitb
from turtle import shape; #cgitb.enable()
from PIL import Image
import cv2
import numpy as np
from improved_sift import improved_sift

form = cgi.FieldStorage()
# Get filename here.
area = form['area_filename']
pieces = form['pieces_filename']

# Test if the file was uploaded
if area.filename and pieces.filename:
   # strip leading path from file name to avoid
   # directory traversal attacks
   area_fn = os.path.basename(area.filename.replace("\\", "/"))
   pieces_fn = os.path.basename(pieces.filename.replace("\\", "/"))
   open(area_fn, 'wb').write(area.file.read())
   open(pieces_fn, 'wb').write(pieces.file.read())
   area_img = cv2.imread(area_fn)
   area_img = np.asarray(area_img)
   pieces_img = cv2.imread(pieces_fn)
   pieces_img = np.asarray(pieces_img)
   #piece = np.uint8(piece)
   #template = np.uint8(template)
   # piece_img_fn = os.path.basename(piece_img.filename.replace("\\", "/"))
   _,_,_,_,sift_result = improved_sift(area_img, pieces_img)
   Image.fromarray(sift_result).convert("RGB").save('result_img.jpg')
   message = 'The files "' + area_fn + '" and "' + pieces_fn + '" were uploaded successfully'
 
else:
   message = 'No file was uploaded'
 
print("Content-Type: text/html")
print("")
print("""
<html>
<style>
  body {background: #E5AB45;
        font-family: 'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif;
        box-sizing: border-box;}
  h1 {text-align: center;
      margin: 30px;}
  img {max-width: 50vw;
       margin: auto;
       display: flex;
       justify-content: center;}
</style>
<body>
   <h1>Solved Puzzle</h1>
   <img src="%s">
</body>
</html>
""" % ("result_img.jpg"))