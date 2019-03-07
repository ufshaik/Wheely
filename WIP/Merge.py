# USAGE
# python distance_to_camera.py

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2 as cv
import argparse
import sys
import os.path

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Wheely')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (5, 5), 0)
	edged = cv.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


# Get the names of the output layers
def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

	label = '%.2f' % conf

	# Get the label for the class name and its confidence
	if classes:
		assert (classId < len(classes))
		label = '%s:%s' % (classes[classId], label)

	# Display the label at the top of the bounding box
	labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
				 (255, 255, 255), cv.FILLED)
	''' To get the name and detection '''
	print(label)
	cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
winName = 'Wheely'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "wheely_out.avi"
if (args.image):
	# Open the image file
	if not os.path.isfile(args.image):
		print("Input image file ", args.image, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.image)
	outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif (args.video):
	# Open the video file
	if not os.path.isfile(args.video):
		print("Input video file ", args.video, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.video)
	outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
	# Webcam input
	cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
								(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

	# get frame from the video
	hasFrame, frame = cap.read()

	# Stop the program if reached end of video
	if not hasFrame:
		print("Done processing !!!")
		print("Output file is stored as ", outputFile)
		cv.waitKey(3000)
		# Release device
		cap.release()
		break

	# Create a 4D blob from a frame.
	blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

	# Sets the input to the network
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers
	outs = net.forward(getOutputsNames(net))

	# Remove the bounding boxes with low confidence
	postprocess(frame, outs)

	# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
	cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

	# Write the frame with the detection boxes
	if (args.image):
		cv.imwrite(outputFile, frame.astype(np.uint8));
	else:
		vid_writer.write(frame.astype(np.uint8))

	cv.imshow(winName, frame)

	# initialize the known distance from the camera to the object, which
	# in this case is 24 inches
	KNOWN_DISTANCE = 24.0

	# initialize the known object width, which in this case, the piece of
	# paper is 12 inches wide
	KNOWN_WIDTH = 11.0

	# load the furst image that contains an object that is KNOWN TO BE 2 feet
	# from our camera, then find the paper marker in the image, and initialize
	# the focal length
	cap = cv.VideoCapture(0)
	ret, frame = cap.read()
	sample = cv.imwrite('C:/Users/ddarkreaper/OneDrive/Work - MetricsFlow/Wheely/sample_images/i.jpg', frame)
	# image = cv2.imread("images/2ft.png")
	marker = find_marker(frame)
	focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

	# loop over the images
	for imagePath in sorted(paths.list_images("sample_images")):
		# load the image, find the marker in the image, then compute the
		# distance to the marker from the camera
		image = cv.imread(imagePath)
		marker = find_marker(image)
		inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

		# draw a bounding box around the image and display it
		box = cv.cv.BoxPoints(marker) if imutils.is_cv2() else cv.boxPoints(marker)
		box = np.int0(box)
		cv.drawContours(image, [box], -1, (0, 255, 0), 2)
		cv.putText(image, "%.2fft" % (inches / 12),
					(image.shape[1] - 200, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
					2.0, (0, 255, 0), 3)
		cv.imshow("image", image)
		cv.waitKey(0)