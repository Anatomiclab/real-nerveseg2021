import os
from PIL import Image
import numpy as np
import shutil
from pathlib import Path

files = os.listdir(".")
for f in files:
    if(f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpg.png')):
        print(f)
        basewidth = 512
        ratio = 1080/1608
        img = Image.open(f)
        img = img.crop((340*ratio,0*ratio,1608*ratio,1608*ratio))
        img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
        img.save(f)
