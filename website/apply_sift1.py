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
piece = form['piece_filename']
template = form['template_filename']
# Test if the file was uploaded
if piece.filename and template.filename:
   # strip leading path from file name to avoid
   # directory traversal attacks
   piece_fn = os.path.basename(piece.filename.replace("\\", "/"))
   template_fn = os.path.basename(template.filename.replace("\\", "/"))
   open(piece_fn, 'wb').write(piece.file.read())
   open(template_fn, 'wb').write(template.file.read())
   piece_img = cv2.imread(piece_fn)
   piece_img = np.asarray(piece_img)
   template_img = cv2.imread(template_fn)
   template_img = np.asarray(template_img)
   #piece = np.uint8(piece)
   #template = np.uint8(template)
   # piece_img_fn = os.path.basename(piece_img.filename.replace("\\", "/"))
   _,_,_,_,sift_result = improved_sift(piece_img, template_img)
   Image.fromarray(sift_result).convert("RGB").save('result_img.jpg')
   message = 'The files "' + piece_fn + '" and "' + template_fn + '" were uploaded successfully'
 
else:
   message = 'No file was uploaded'
 
print("Content-Type: text/html")
print("")
print("""
<html>
<style>
  body {background: #588985;
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