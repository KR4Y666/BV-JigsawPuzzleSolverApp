#! C:\Python\Python310\python.exe
import cgi, os
import cgitb
from turtle import pu; #cgitb.enable()
from PIL import Image
import cv2
import numpy as np
from jigsaw_puzzle_solver import puzzle_solver

form = cgi.FieldStorage()
# Get filename here.
fileitem = form['filename']
# Test if the file was uploaded
if fileitem.filename:
   # strip leading path from file name to avoid
   # directory traversal attacks
   fn = os.path.basename(fileitem.filename.replace("\\", "/"))
   open(fn, 'wb').write(fileitem.file.read())
   img = cv2.imread(fn)
   img = np.asarray(img)
   result = puzzle_solver(fn)
   Image.fromarray(result).convert("RGB").save('solver_result_img.jpg')


   message = 'The file "' + fn + '" was uploaded successfully'
 
else:
   message = 'No file was uploaded'
 
print("Content-Type: text/html")
print("")
print("""
<html>
<style>
  body {background: #D18A8A;
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
""" % ("solver_result_img.jpg"))
