import os
import glob
import imageio
from natsort import natsorted, ns
import cv2
import numpy as np

png_dir = ''

images = []
file_names = glob.glob("*.png")
print('before sort',file_names)
file_names = natsorted(file_names, key=lambda y: y.lower())
print('sorted',file_names)

name = os.path.dirname(os.path.realpath(__file__))

for file_name in file_names:
    img = cv2.imread(file_name)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    print(file_name)

out = cv2.VideoWriter(name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

for i in range(len(images)):
    out.write(images[i])
out.release()
