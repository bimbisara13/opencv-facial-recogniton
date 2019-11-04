# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle \
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import dlib
import openface

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["./face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["./face_detection_model",
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("./face_detection_model/openface_nn4.small2.v1.t7")
predictor_model = "./face_detection_model/shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

imagepath = "./dataset/bhavika/000011.jpg"
image = cv2.imread(imagepath)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)
detector.setInput(imageBlob)
detections = detector.forward()
if len(detections) > 0:
    # we're making the assumption that each image has only ONE
    # face, so find the bounding box with the largest probability
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    # ensure that the detection with the largest probability also
    # means our minimum probability test (thus helping filter out
    # weak detections)
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for
        # the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI and grab the ROI dimensions
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        alignedFace = face_aligner.align(534, face, face_aligner.getLargestFaceBoundingBox(face), landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        cv2.imshow("Frame1", face)
        cv2.imshow("Frame", alignedFace)
        cv2.waitKey(0)
