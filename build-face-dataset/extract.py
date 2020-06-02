from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

vs = cv2.VideoCapture("/Users/Sai/Desktop/Facial-Recognition/Videos/yes.mp4")
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 0

try:

	# creating a folder named data
	if not os.path.exists('data'):
		os.makedirs('data')

# if not created then raise error
except OSError:
	print ('Error: Creating directory of data')

# frame
currentframe = 1

# loop over the frames from the video stream
while True:

	ret,frame = vs.read()
	if ret:
		name = './data/' + str(currentframe) + '.jpg'
		print ('Creating...' + name)
		frame = cv2.resize(frame,(300,300))
		M = cv2.getRotationMatrix2D((300/2,300/2), 90, 1)
		frame = cv2.warpAffine(frame, M, (300,300))
		cv2.imwrite(name, frame)
		currentframe += 1
	else:
		break

# Release all space and windows once done
vs.release()
cv2.destroyAllWindows()
