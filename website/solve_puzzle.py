#! C:\Python\Python310\python.exe

# imports
import cgi, os
from PIL import Image
import cv2
import numpy as np
from jigsaw_puzzle_solver import puzzle_solver

# get fieldstorage
form = cgi.FieldStorage()

# get filename
fileitem = form['filename']
piece_number = int(form['piece_number'].value)

# test if the file was successfully uploaded
if fileitem.filename:
   
   # get uploaded image
   fn = os.path.basename(fileitem.filename.replace("\\", "/"))
   open(fn, 'wb').write(fileitem.file.read())
   img = cv2.imread(fn)
   img = np.asarray(img)

   # solve puzzle
   result = puzzle_solver(fn, piece_number)
   Image.fromarray(result).convert("RGB").save('solver_result_img.jpg')

   message = ""
 
else:
   message = 'No files were uploaded'
 
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
   <p>%s</p>
</body>
</html>
""" % ("solver_result_img.jpg", message))
