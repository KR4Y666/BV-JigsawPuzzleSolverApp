#! C:\Python\Python310\python.exe

# imports
import cgi, os
from PIL import Image
import cv2
import numpy as np
from improved_sift import improved_sift

# get fieldstorage
form = cgi.FieldStorage()

# get filenames
area = form['area_filename']
pieces = form['pieces_filename']

# test if the files were successfully uploaded
if area.filename and pieces.filename:
   
   # get uploaded images
   area_fn = os.path.basename(area.filename.replace("\\", "/"))
   pieces_fn = os.path.basename(pieces.filename.replace("\\", "/"))
   open(area_fn, 'wb').write(area.file.read())
   open(pieces_fn, 'wb').write(pieces.file.read())
   area_img = cv2.imread(area_fn)
   area_img = np.asarray(area_img)
   pieces_img = cv2.imread(pieces_fn)
   pieces_img = np.asarray(pieces_img)

   # use imporved_sift function get result image which puzzle piece belongs to the area
   _,_,_,_,sift_result = improved_sift(area_img, pieces_img)
   Image.fromarray(sift_result).convert("RGB").save('result_img.jpg')
   message = ""
 
else:
   message = 'No files were uploaded'
 
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
   <p>%s</p>
</body>
</html>
""" % ("result_img.jpg", message))