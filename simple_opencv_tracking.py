import cv2
from sys import exit
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# Set up tracker.
# Instead of MIL, you can also use
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	#"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	#"tld": cv2.TrackerTLD_create,
	#"medianflow": cv2.TrackerMedianFlow_create,
	#"mosse": cv2.TrackerMOSSE_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# Read video
video = cv2.VideoCapture(args["video"])

# Exit if video not opened.
if not video.isOpened():
    exit("Could not open video")

# Read first frame.
ok, frame = video.read()
if not ok:
    exit('Cannot read video file')

# Define an initial bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
    	# Tracking success
		#(x, y, w, h) = bbox
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #cv2.rectangle(frame, (x, y), (x + w, y + h),
		#			  (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected",
					(100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, args["tracker"] + " Tracker",
				(100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)),
				(100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
