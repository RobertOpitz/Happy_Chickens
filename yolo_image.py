import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
                help = "path to input image")
ap.add_argument("-y", "--yolo", required = True,
                help = "base path to YOLO directory")
ap.add_argument("-c", "--confidence", type = float, default = 0.5,
                help = "minimum probability to filter weak detection")
ap.add_argument("-t", "--threshold", type = float, default = 0.3,
                help = "threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
#print(args)

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = "uint8")
# COLORS is a numpy object, but later we need a standard list, so we convert the
# numpy array to a standard list object.
COLORS = COLORS.tolist()

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolo-coco.weights"])
configPath = os.path.sep.join([args["yolo"], "yolo.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk ...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
# de-facto: Pre-Processing of one image
blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416),
                             swapRB = True, crop = False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialze our lists of detected bounding boxes, confidences, and class IDs
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of thedetections
    for detection in output:
        # extract the class ID and confidence (aka probability) of the current
        # object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected probability is
        # greater than the minim probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back realtive to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x,y)-coordinates of the bounding box
            # followed by the boxes width and height
            box = detection[0:4] * np.array([W, H, W, H])
            # the object box has the content:
            # [centerX, centerY, width, height]

            # Use the center (x,y)-coordinates to derive the bottom left
            # corner of the bounding box. This replaces the old coordinates
            # of the center of the box.
            box[0:2] = box[0:2] - 0.5 * box[2:5]

            # add the found box, confidencen and classID to our list of
            # bounding box coordinates, confidences, and class IDs
            boxes.append(box.astype("int"))
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences,
                        args["confidence"], args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        # (x, y) is the bottom left corner of the rectangle
        # (w, h) is the width and height of the box
        (x, y, w, h) = boxes[i]

        # draw a bouding box rectangle and label on the image
        color = COLORS[classIDs[i]]
        cv2.rectangle(image, (x,y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

# show the output image
cv2.namedWindow("Image")
cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
