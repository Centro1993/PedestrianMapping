# import the necessary packages
import argparse
from webcam import Webcam
import numpy as np
import math
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-i", "--image", help="path to the background image")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-c", "--calibration-time", type=int, default=300, help="time between calibrations")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam.py
if args.get("video", None) is None:
    camera = Webcam(0)
    camera.start()
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = Webcam(args["video"])
    camera.start()

# load mask image
backgroundImg = cv2.imread(args["image"]).astype(np.float)
backgroundImgHeight, backgroundImgWidth, _ = backgroundImg.shape

# load calibration image
calibrationImg = cv2.imread("chessboard.png")
# resize for fullscreen
calibrationImg = cv2.resize(calibrationImg, (backgroundImgWidth, backgroundImgHeight))

'''
CALIBRATION PHASE
'''

def get_projection_distortion(sampleCount):
    print "Getting Projection Distortion..."

    # count successful samples
    i = 0

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # show calibration image in fullscreen
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", calibrationImg)

    while i < sampleCount:

        # get image from webcam.py
        image = camera.get_current_frame()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # DEBUG display image for setup
        # cv2.imshow("test", image)

        # look for chessboard in current pic
        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 6), None)

        if ret:
            i += 1

            print str(i) + " out of " + str(sampleCount) + " Calibration Images done"

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # image = cv2.drawChessboardCorners(image, (7, 6), corners, ret)

        # if the `q` key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    return objpoints, imgpoints, gray


def get_projection_corners(imgpoints, shape):
    print "Getting Projection Corners..."

    # print imgpoints
    rect = np.zeros((4, 2), dtype="float32")

    for ip, point in enumerate(rect):
        for ic, coord in enumerate(point):
            rect[ip][ic] = 56789098

    for image in imgpoints:
        for point in image:
            point = point[0]

            # find top-left corner
            if math.hypot(point[0], point[1]) < math.hypot(rect[0][0], rect[0][1]):
                rect[0] = point

            # find top-right corner
            if math.hypot(point[0] - shape["width"], point[1]) < math.hypot(rect[1][0] - shape["width"], rect[1][1]):
                rect[1] = point

            # find lower-right corner
            if math.hypot(point[0] - shape["width"], point[1] - shape["height"]) < math.hypot(rect[2][0] - shape["width"], rect[2][1] - shape["height"]):
                rect[2] = point

            # find lower-right corner
            if math.hypot(point[0], point[1] - shape["height"]) < math.hypot(rect[3][0], rect[3][1] - shape["height"]):
                rect[3] = point

    return rect


def four_point_transform(img, rect):
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def calibration():
    print "Calibrating..."
    cv2.destroyAllWindows()
    objpoints, imgpoints, gray = get_projection_distortion(3)
    height, width = gray.shape
    rect = get_projection_corners(imgpoints, {"width": width, "height": height})
    print rect

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # destroy all windows, display black image and wait, so we have a neutral first frame
    cv2.destroyAllWindows()
    blackImg = np.zeros((backgroundImgHeight, backgroundImgWidth, 3), np.float)
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Calibration", blackImg)
    cv2.waitKey(3000)
    time.sleep(3)
    cv2.destroyAllWindows()

    return rect


'''
MOTION DETECTION
'''

#global variables
calibration_timestamp = time.time()

rect = calibration()

# initialize the first frame and save the last output frame in the video stream
firstFrame = None
lastOutput = None

# save starting timestamp
lastTimestamp = time.time()

# loop over the frames of the video
while True:

    # calibrate after a set time
    if calibration_timestamp < time.time() - args["calibration_time"]:
        rect = calibration()
        firstFrame = None
        lastOutput = None
        calibration_timestamp = time.time()

    # grab the current frame
    frame = camera.get_current_frame()

    # apply the transform to crop out and warp the projection part of the image
    frame = four_point_transform(frame, rect)
    cv2.imshow("Debug", frame)

    # if lastOutput is not None:
    #    frame = cv2.subtract(frame, lastOutput)

    # convert it to grayscale, and blur it
    # frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)


    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        # close black frame
        cv2.destroyAllWindows()
        continue

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # generate black base image
    height, width, _ = frame.shape
    black = np.zeros((height, width, 3), np.float)

    # loop over the contours
    for i, c in enumerate(cnts):
        # if the contour is too small, fade it out
        alphaMultiplicator = cv2.contourArea(c) / args["min_area"]

        cv2.drawContours(black, cnts, i, (alphaMultiplicator, alphaMultiplicator, alphaMultiplicator), 5)

    # apply postproduction
    black = cv2.GaussianBlur(black, (31, 31), 4)

    # move background image around
    centerX = backgroundImgWidth / 2
    centerY = backgroundImgHeight / 2

    backgroundImgX = int(centerX + (np.cos(lastTimestamp-time.time() % 2) * (centerY / 2)))
    backgroundImgY = int(centerY + (np.cos(lastTimestamp-time.time() % 2) * (centerX / 2)))

    backgroundImgX = min(backgroundImgX, backgroundImgWidth - backgroundImgX)
    backgroundImgY = min(backgroundImgY, backgroundImgHeight - backgroundImgY)

    backgroundImgTrimmed = backgroundImg[backgroundImgY:height + backgroundImgY, backgroundImgX:width + backgroundImgX]

    output = backgroundImgTrimmed * black / 255


    # show the computed frame
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Output", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Output", output)

    # DEBUG show unmodified image for debug
    # cv2.imshow("Debug", frame)

    # save timestamp
    lastTimestamp = time.time()

    # save output
    lastOutput = output

    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)

    # if the `q` key is pressed, break from the lop
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# cleanup the camera and close any open windows
cv2.destroyAllWindows()
exit(0)