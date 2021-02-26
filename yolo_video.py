import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
                help = "path to input video")
ap.add_argument("-o", "--output", required = True,
                help = "path to output video")
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

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(ln)

# initialize the video stream (vs), and frame dimensions
vs = cv2.VideoCapture(args["input"])
(W, H) = int(vs.get(3)), int(vs.get(4))

# initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

# determine the total number of frames in the video file
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
print("[INFO] {} total frames in video".format(total))

# collecting the total time as the sum of time needed for processing one frame
total_time = 0.0

# loop over frames from the video file stream
for j in range(1, total+1):

    # start time for processing one frame
    start = time.time()

    # GET THE NEXT FRAME FROM THE VIDEO
    (grapped, frame) = vs.read()

    # if the frame was not grapped, then we have reached the end of the stream
    if not grapped:
        print("[ERROR] Could not read frame. Program is terminated")
        break

    # PRE-PROCESSING OF THE FRAME
    # construct a blob from the image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416),
                                 swapRB = True, crop = False)
    net.setInput(blob)
    # DO THE PREDICTION
    layerOutputs = net.forward(ln)

    # GET THE RESULTS FROM THE PREDICTIONS
    # initialze our lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # The detection object is a vector.
            # The first 4 elemenst is the box of the assumed detected object.
            # The fith element seems to be corrletaed to the predicted
            # probabilities, but is at the moment unknown.
            # The elements starting with the 6th element are the probabilities
            # of a specific object class (here of coco, so 80 classes).
            # Extract the class ID with the highest confidence (aka probability)
            # of the current object detection together with the box and the
            # confidence for that object.

            # theses are the probabilities of all predicted classes
            scores = detection[5:]
            # get the ID of the class with the highest probability
            classID = np.argmax(scores)
            # get the highest probability
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimal allowed probability
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

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes.
    # This function returns the index of all important boxes as a numpy array.
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,
                            args["confidence"], args["threshold"])

    # PLOT RECTANGLES AROUND THE FOUND OBJECTS WITH NAMES AND STUFF
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        # the index is numpy array, and is transfomred now into a simple vector
        for i in idxs.flatten():
            # extract the bounding box coordinates
            # (x, y) is the bottom left corner of the rectangle
            # (w, h) is the width and height of the box
            (x, y, w, h) = boxes[i]

            # draw a bounding box rectangle and label on the image
            color = COLORS[classIDs[i]]
            cv2.rectangle(frame, (x,y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # WRITE THE TREATED FRAME TO THE OUTPUT FILE
    writer.write(frame)

    # get the elapsed time for one frame
    # this is for printing some information for the user
    elapsed_time = time.time() - start
    total_time += elapsed_time

    # print progress bar, percentage of work done, time to process one frame,
    # and time left until all frames are processed
    print("\r[INFO] Progress: [{0:20s}] {1:.1f}% ; {2:.2f} sec per frame ; "
          "{3:.2f} min left ".format('#' * round(20 * j/total),
                                     100 * j / total,
                                     elapsed_time,
                                     (total - j) * total_time / (60 * j)),
          end = "", flush = True)


# release the file pointer
print("\n[INFO] cleaning up ...")
writer.release()
vs.release()
