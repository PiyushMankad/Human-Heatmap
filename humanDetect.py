
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from logfile import createLog


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.02,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
CLASSES=["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

vs = VideoStream(src="C:\\Users\\Ayush\\Videos\\DJI_0012.MOV").start()

# FOR LIVE STREAMING
#vs = VideoStream(src="rtmp://10.1.1.1/live/bird").start()


# loop over the frames from the video stream
startTime=time.time()
while True:
	currentTime=time.time()
	peopleCount=0
	frame = vs.read()
	frame=cv2.resize(frame,(1280,720))
	#frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]

	if currentTime-startTime>5:
		startTime=time.time()

		blob = cv2.dnn.blobFromImage(frame,0.007843, (1280, 720), 127.5)
		net.setInput(blob)
		detections = net.forward()
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				if idx==15:
					peopleCount+=1
					label = "{}: {:.2f}%".format(CLASSES[0],
						confidence * 100)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						COLORS[0], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
					print(" No of people- ",peopleCount)

	# show the output frame
		cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
