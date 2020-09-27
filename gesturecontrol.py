# import the necessary packages
from imutils.video import FPS
from yoloObjectDetection.ObjectDetector import *
import argparse
import time
import cv2
import threading
import csv
import sys

class GestureControl:
    def __init__(self, trackerType = "csrt"):
        self.trackerType = trackerType
        self.running = False
        self.vectorLock = threading.Lock()
        self.objectDetector = ObjectDetector(0.25, 0.4, 80, 160)
        self.runThroughDetectorCount = 0

        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        self.tracker = OPENCV_OBJECT_TRACKERS[trackerType]()
        self.vectorX = 0.0
        self.vectorY = 0.0
        self.vectorZ = 0.0

    def grabVector(self):
        vectorX = 0.0
        vectorY = 0.0
        vectorZ = 0.0

        # Acquire lock, read the values then release lock
        if self.vectorLock.acquire():
            vectorX = self.vectorX
            vectorY = self.vectorY
            vectorZ = self.vectorZ
            self.vectorLock.release()

        return (vectorX, vectorY, vectorZ)

    def updateVectors(self, vectorX, vectorY, vectorZ):
        if self.vectorLock.acquire(False):
            self.vectorX = vectorX
            self.vectorY = vectorY
            self.vectorZ = vectorZ
            self.vectorLock.release()
            return True

        return False

    def stream(self, saveData = False):
        # initialize the bounding box coordinates of the object we are going
        # to track
        initBB = None

        # if a video path was not supplied, grab the reference to the web cam
        print("[INFO] starting video stream...")
        self.running = True
        cap = cv2.VideoCapture(0)
        time.sleep(1.0)

        # Initialise variables
        fps = None
        preX = 0.0
        preY = 0.0
        preZ = 0.0

        vectorX = 0.0
        vectorY = 0.0
        vectorZ = 0.0

        vectorDataSet = []
        startTime = time.time()
        elapsedTimeSinceFirstDetection = 0
        # loop over frames from the video stream
        while cap.isOpened():
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture objectq
            ret, frame = cap.read()

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            # resize the frame (so we can process it faster) and grab the
            # frame dimensions
            # frame = cv2.resize(frame, (320, 320))
            (H, W) = frame.shape[:2]
            print(H, W)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Check if we have detected initial bounding box or how many times we ran it through the detector
            # The detector count is meant to refine the bounding box location
            if initBB == None:
                initBB = self.objectDetector.detectObject(ret, frame)
                if initBB != None:
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    # self.tracker.init(frame, initBB)
                    self.tracker.init(frame, initBB)
                    fps = FPS().start()
                    print(initBB)
                    startTime = time.time()
                    cv2.rectangle(frame, tuple(initBB[0:2]), tuple(initBB[2:4]), (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)

                else:
                    continue


            # check to see if we are currently tracking an object
            if initBB is not None:
                print("Tracking object")
                # grab the new bounding box coordinates of the object

                (success, box) = self.tracker.update(frame)
                print(box)
                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    vectorX += x + w/2 - preX
                    vectorY += y + h/2 - preY
                    vectorZ += 0.0 # Needs work
                    preX = x + w/2
                    preY = y + h/2

                    # If unable to acquire the lock, don't update the vectors
                    if self.updateVectors(vectorX, vectorY, vectorZ) is True:
                        vectorX = 0.0
                        vectorY = 0.0
                        vectorZ = 0.0

                    elapsedTimeSinceFirstDetection = time.time() - startTime
                    if writecsv:
                        data = list(self.grabVector())
                        data.append(elapsedTimeSinceFirstDetection)
                        vectorDataSet.append(data)

                # update the FPS counter
                fps.update()
                fps.stop()

                # initialize the set of information we'll be displaying on
                # the frame
                info = [
                    ("Tracker", self.trackerType),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                    ("Centre of Box", "{}: {}".format(x + w/2, y + h/2)),
                    ("Vector of Box", "{:.2f}, {:.2f}, {:.2f}".format(self.vectorX, self.vectorY, self.vectorZ)),
                ]

                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Frame", frame)

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if we are using a webcam, release the pointer
        self.running = False

        # close all windows
        cv2.destroyAllWindows()

        if writecsv:
            self.createCSV(vectorDataSet)

    def playback(self, fileName):
        dataList = self.getCSVData(fileName)
        if dataList == []:
            print("File is empty")
            sys.exit()

        print("[INFO] Playback data from csv sheet")
        self.running = True
        while self.checkRunning():
            previousDataSetTime = 0
            for data in dataList:
                if data == dataList[0]:
                    sleepTime = data[3]
                else:
                    sleepTime = data[3] - previousDataSetTime
                vectorX = data[0]
                vectorY = data[1]
                vectorZ = data[2]
                time.sleep(sleepTime)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    # Needs work
                    self.running = False
                    break

                self.updateVectors(vectorX, vectorY, vectorZ)

                previousDataSetTime = data[3]

            self.running = False

        cv2.destroyAllWindows()

    # Returns the state of gesture control
    def checkRunning(self):
        return self.running

    def createCSV(self, dataSet):
        with open('GesturePlayBack.csv', 'w', newline ='') as file:
            writer = csv.writer(file)
            for set in dataSet:
                writer.writerow(list(set))

    def getCSVData(self, filename):
        data = []

        try :
            with open(filename) as csvFile:
                reader = csv.reader(csvFile, delimiter = ',')
                for row in reader:
                    if row != []:
                        data.append([float(item) for item in row])
            csvFile.close()
        except FileNotFoundError:
            print("No file exists with the name", filename)
            sys.exit()

        return data

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--playback", type=str, default="",
        help="Specify playback file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
        help="OpenCV object tracker type")
    ap.add_argument("-w", "--writecsv", type=str, default="", help="write -> writes a csv")
    args = vars(ap.parse_args())

    writecsv = args['writecsv'] == "write"
    playback = args['playback'] != ''

    gestureControl = GestureControl(args["tracker"])

    if playback:
        gestureControlThread = threading.Thread(target = gestureControl.playback, args=([args['playback']]))
    else:
        gestureControlThread = threading.Thread(target = gestureControl.stream, args=([writecsv]))
    gestureControlThread.setDaemon = True
    gestureControlThread.start()

    # time.sleep(0.2)
    # while gestureControl.checkRunning():
    #     time.sleep(0.05)
    #     print(gestureControl.grabVector())

