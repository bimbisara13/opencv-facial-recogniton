import cv2
import numpy
from os import listdir
from os.path import isfile, join

mypath = '/Users/Sai/Downloads/Photos'
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

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )

        print("[INFO] Found {0} Faces.".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = images[i][y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite('./data/' + str(k) + '_faces.jpg', roi_color)
            k = k+ 1
