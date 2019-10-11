# USAGE
# python build_face_dataset.py --cascade haarcascade_frontalface_default.xml --output dataset/adrian

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
from fastai.vision import *
import torch
from fastai.metrics import *
#for facenet call below line
from facenet_pytorch import InceptionResnetV1

path = "/Users/Sai/Desktop/Facial-Recognition/build-face-dataset/"
learn = load_learner(path, 'facenet.pkl')

classes = ['aaisha', 'anurag', 'ashish', 'bhavika', 'prershen', 'sam', 'simran']


# construct the argument parser and parse the arguments


pred_class_true = "detecting"
# load OpenCV's Haar cascade for face detection from disk
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0
f = 0
average = [0,0,0,0,0,0,0]
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	f = f + 1
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3,
		minNeighbors=5, minSize=(30, 30))

	# loop over the face detections and draw them on the frame and predict face
	for (x, y, w, h) in rects:
		image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = image[y:y + h, x:x + w]
		pred_class = learn.predict(Image(pil2tensor(roi_color, np.float32).div_(255)))
		average = average[0] + pred_class[2]
		print(f)
		if (f % 5 == 0):
			print(average)
			average = average/5
			max = 0
			maxi = 0
			for i in range(7):
				if(max < average[i]):
					max = average[i]
					maxi = i
				pred_class_true = classes[maxi]
			f = 0
			average = [0,0,0,0,0,0,0]
		cv2.putText(image, str(pred_class_true), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
		#p = os.path.sep.join([args["output"], "{}.png".format(
			#str(total).zfill(5))])
		#cv2.imwrite(p, orig)
		total += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break


# do a bit of cleanup
print("closing")
cv2.destroyAllWindows()
vs.stop()
