import glob
import os
from PIL import Image

types = ("*.jpg","*.png")
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(files))
f= open("filename.txt","w+")
for image in files_grabbed:
    f.write(os.path.splitext(os.path.basename(image))[0]+"\n")

f.close()
