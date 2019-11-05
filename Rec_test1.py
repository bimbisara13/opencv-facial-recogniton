# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle sklearn.utils._cython_blas

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import dlib
import openface

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("./face_detection_model/openface_nn4.small2.v1.t7")

predictor_model = "./face_detection_model/shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("./output/recognizer.pickle", "rb").read())
le = pickle.loads(open("./output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame,+1)
	(h, w) = frame.shape[:2]

	face = face_aligner.getLargestFaceBoundingBox(frame)
	x = face.left()
	y = face.top()
	w = face.width()
	h = face.height()


	facee = frame[y:y+h, x:x+w]
	alignedFace = face_aligner.align(534, facee,face_aligner.getLargestFaceBoundingBox(facee), landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
	try:
		faceBlob = cv2.dnn.blobFromImage(alignedFace, 1.0 / 255,
			(96, 96), (0, 0, 0), swapRB=True, crop=False)
	except:
		pass
	embedder.setInput(faceBlob)
	vec = embedder.forward()

			# perform classification to recognize the face
	preds = recognizer.predict_proba(vec)[0]
	j = np.argmax(preds)
	proba = preds[j]
	name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
	text = "{}: {:.2f}%".format(name, proba * 100)
	cv2.rectangle(frame ,(x,y),(x+w,y+h),(255, 0, 0), 2)
	cv2.putText(frame, text, (x, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
