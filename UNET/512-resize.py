import os
from PIL import Image
import numpy as np
import shutil
from pathlib import Path

files = os.listdir(".")
for f in files:
    if(f.endswith('.jpg') or f.endswith('.png')):
        print(f)
        basewidth = 512
        img = Image.open(f)
        #img = img.crop((500,160,1574,1388))
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
        img.save(f)
