#! C:\Python\Python310\python.exe

# imports
import cgi, os
from PIL import Image
import cv2
import numpy as np
from improved_sift import improved_sift, search_position


# get fieldstorage
form = cgi.FieldStorage()

# get filenames and number of pieces
piece = form['piece_filename']
template = form['template_filename']
piece_number_horizontal = int(form['piece_number_horizontal'].value)
piece_number_vertical = int(form['piece_number_vertical'].value)

# test if the files were successfully uploaded
if piece.filename and template.filename:

   # get uploaded images
   piece_fn = os.path.basename(piece.filename.replace("\\", "/"))
   template_fn = os.path.basename(template.filename.replace("\\", "/"))
   open(piece_fn, 'wb').write(piece.file.read())
   open(template_fn, 'wb').write(template.file.read())
   piece_img = cv2.imread(piece_fn)
   piece_img = np.asarray(piece_img)
   template_img = cv2.imread(template_fn)
   template_img = np.asarray(template_img)

   # if user knows the number of horizontal and vertical pieces, use search_position function for
   # better result image
   if piece_number_horizontal > 0 and piece_number_vertical > 0:
      sift_result = search_position(piece_img, template_img, piece_number_horizontal, piece_number_vertical)
      Image.fromarray(sift_result).convert("RGB").save('result_img.jpg')

   # if user doesn't know the number of horizontal and vertical pieces, use the normal improved_sift function
   else:
      _,_,_,_,sift_result = improved_sift(piece_img, template_img)
      Image.fromarray(sift_result).convert("RGB").save('result_img.jpg')
      
   message = ""
 
else:
   message = 'No files were uploaded'
 
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
   <p>%s</p>
</body>
</html>
""" % ("result_img.jpg", message))