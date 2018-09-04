
# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
from carlog import createLog
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default=r"E:\Open CV\K steps\images\street.jpg",
	help="path to input image")
ap.add_argument("-p", "--prototxt",default="MobileNetSSD_deploy.prototxt.txt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

half=int(w/2)
leftcoords=[]		
rightcoords=[]
middlecoords=[]
lmax=0.0		#probability of left object
rmax=0.0		#probability of right object

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
print("width",w)
net.setInput(blob)
detections = net.forward()

#	#	#	detections(_,index,confidence,startx,starty,endx,endy)	#	#	#
#print(detections)
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		
		if idx==7 or idx==6:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			if startX<half and endX<half:
				if confidence>lmax:
					lmax=confidence
					if lmax>0.80:

						leftcoords=[startX,startY,endX,endY]
						print("left  ",leftcoords,"conf-",lmax)
						print(len(leftcoords))
						
						# display the prediction
						label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						print("[INFO] {} {} {} {} {}".format(label,startX,startY,endX,endY))
						#createLog("[Left Car] {} {} {} {} {}".format(label,startX,startY,endX,endY))
						cv2.rectangle(image, (startX, startY), (endX, endY),
							COLORS[idx], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(image, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
					

			elif startX>half and endX>half:
				if confidence>rmax:
					rmax=confidence
					if rmax>0.80:

						rightcoords=[startX,startY,endX,endY]
						print("right  ",rightcoords,"conf-",rmax)
						print(len(rightcoords))

						# display the prediction
						label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						print("[INFO] {} {} {} {} {}".format(label,startX,startY,endX,endY))
						#createLog("[Right Car] {} {} {} {} {}".format(label,startX,startY,endX,endY))
						cv2.rectangle(image, (startX, startY), (endX, endY),
							COLORS[idx], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(image, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			else:
				middlecoords.append([startX,startY,endX,endY])
			'''
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				print("[INFO] {} {} {} {} {}".format(label,startX,startY,endX,endY))
				#createLog("[INFO] {} {} {} {} {}".format(label,startX,startY,endX,endY))
				cv2.rectangle(image, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			'''


### distance calculation of free space between cars ####
distance=0
if len(leftcoords)==4 and len(rightcoords)==4:
	distance=int(rightcoords[0]-leftcoords[2])
	y=leftcoords[3] if leftcoords[3]>rightcoords[3] else rightcoords[3]
	cv2.line(image,(rightcoords[0],y),(leftcoords[2],y),(255,0,0),2)
	print("Distance b/w cars {} pixels and depth {} pixels".format(distance,(h-y)))
	#createLog("Distance b/w cars {} pixels and depth {} pixels".format(distance,y))

elif len(leftcoords)==4:
	if len(middlecoords)!=0:
		distance=startX
		y=endY
		cv2.line(image,(0,y),(startX,y),(255,0,0),2)
		print("Distance b/w cars {} pixels and depth {} pixels".format(distance,(h-y)))
		print("1")
		#createLog("Distance b/w cars {} pixels and depth {} pixels".format(distance,endY))

	else:
		distance=w-endX
		y=endY
		cv2.line(image,(w,y),(endX,y),(255,0,0),2)
		print("Distance b/w cars {} pixels and depth {} pixels".format(distance,(h-y)))
		#createLog("Distance b/w cars {} pixels and depth {} pixels".format(distance,endY))
		print("2")

elif len(rightcoords)==4:
	if len(middlecoords)!=0:
		distance=w-endX
		y=endY
		cv2.line(image,(w,y),(endX,y),(255,0,0),2)
		print("Distance b/w cars {} pixels and depth {} pixels".format(distance,(h-y)))
		#createLog("Distance b/w cars {} pixels and depth {} pixels".format(distance,endY))
		print("3")
	else:
		distance=startX
		y=endY
		cv2.line(image,(0,y),(startX,y),(255,0,0),2)
		print("Distance b/w cars {} pixels and depth {} pixels".format(distance,(h-y)))
		#createLog("Distance b/w cars {} pixels and depth {} pixels".format(distance,endY))
		print("4")

print("middle",len(middlecoords))
# show the output image
cv2.imshow("Output", image)
while True:
	if cv2.waitKey(1) & 0xFF==ord('q'):
		cv2.destroyAllWindows()
		sys.exit()

