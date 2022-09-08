#! C:\Python\Python310\python.exe
import cgi, os
import cgitb; #cgitb.enable()
from improved_sift import search_position

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
   sift_result = search_position(piece_fn, template_fn, 20)
   open(piece_fn, 'wb').write(piece.file.read())
   open(template_fn, 'wb').write(template.file.read())
   message = 'The files "' + piece_fn + '" and "' + template_fn + '" were uploaded successfully'
 
else:
   message = 'No file was uploaded'
 
print("Content-Type: text/html")
print("")
print("""
<html>
<body>
   <p>%s</p>
   <img src="%s">
</body>
</html>
""" % (message, sift_result))