import cv2
import numpy
from os import listdir
from os.path import isfile, join

mypath = '/Users/Sai/Desktop/Facial-Recognition/Faces/train/ashish'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)

k = 1000
print (len(onlyfiles))
for i in range(0,len(onlyfiles)):
    print (i)
    images[i] = cv2.imread( join(mypath,onlyfiles[i]) )
    if(images[i] is None):
        continue
    else:
        gray = images[i].copy()
        cv2.imwrite('./data/' + str(k) + '_faces.jpg',gray)
