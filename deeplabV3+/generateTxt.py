from PIL import Image
import numpy as np
import glob, os

totalList = []
trainRatio = 0.8
f = open('trainval.txt','w')
for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    f.write(file+"\n")
    totalList.append(file)
f.close()
np.random.shuffle(totalList)
print(totalList)
totalNum = len(totalList)
trainNum = round(trainRatio*totalNum)
trainList, valList = totalList[:trainNum], totalList[trainNum:]
f = open('train.txt','w')
for x in trainList:
    f.write(x+"\n")
f.close()
f = open('val.txt','w')
for x in valList:
    f.write(x+"\n")
f.close()
