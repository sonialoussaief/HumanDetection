__author__ = 'Alex'

import argparse
import imutils
import time
import cv2
import freenect
import numpy as np
from cv2 import cv
import dlib


def convert2true_coord(l, t, r, b, s_width=640, t_width=1280):
    rate = t_width/s_width
    return int(l*rate), int(t*rate), int(r*rate), int(b*rate)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default='/Users/Alex/Desktop/HumanDetection/video/Video2.mov', help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
args = vars(ap.parse_args())
drift = 70

# reading from a video file
cap = cv2.VideoCapture(args["video"])
# get the video true width
width_true = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
height_true = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
# initialize hog
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize tracking list
track_list = []
# create x11 window
# win = dlib.image_window()


##################################################
#   Start looping over the frames of the video   #
##################################################
while True:
    # grab the current frame and last frame
    (grabbed1, frame1) = cap.read()
    cap.read()
    cap.read()
    cap.read()
    (grabbed2, frame2) = cap.read()

    # if the frame not grabbed, break
    if not grabbed1:
        break

    # shrink the frame and convert them to gray scale then blur for easier to find the frame difference
    frame1_shrunk = imutils.resize(frame1, width=640)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

    frame2_shrunk = imutils.resize(frame2, width=640)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # BGR --> RGB for X11 window
    # img_RGB = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # compute the absolute difference between the current frame and the former frame
    frameDelta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]

    ##################################################
    #     update the trackers in the track list      #
    ##################################################
    # win.set_image(img_RGB)
    if track_list:
        # win.clear_overlay()
        for t in track_list:
            t.update(gray2)
            p = t.get_position()
            track_t = int(dlib.drectangle.top(p))
            track_l = int(dlib.drectangle.left(p))
            track_b = int(dlib.drectangle.bottom(p))
            track_r = int(dlib.drectangle.right(p))
            # set the thresh area where tracking box located as 0 for not detecting that specific area motion
            thresh[track_t-10:track_b+10, track_l-10:track_r+10] = 0
            # convert shrunk coordinates to true coordinates
            # l_t, t_t, r_t, b_t = convert2true_coord(track_l, track_t, track_r, track_b)
            # draw tracking box in OpenCV window
            # cv2.rectangle(frame2, (l_t, t_t), (r_t, b_t), (0, 255, 0), 2)
            cv2.rectangle(frame2, (track_l, track_t), (track_r, track_b), (0, 255, 0), 2)
            # draw tracking box in X11 window
            # win.add_overlay(t.get_position())

    # dilate the threshold image to fill in holes and find contours on threshold image
    thresh = cv2.dilate(thresh, None, iterations=10)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_filtered = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        if cv2.contourArea(c) > width_true*height_true/8:
            continue
        # get the bounding box for each contour
        (x, y, w, h) = cv2.boundingRect(c)
        # find out the human hog descriptor around(2*drift longer than width & height) the frame_diff areas
        row_start = np.clip(y-drift, 0, gray2.shape[0])
        row_end = np.clip(y+h+drift, 0, gray2.shape[0])
        col_start = np.clip(x-drift, 0, gray2.shape[1])
        col_end = np.clip(x+w+drift, 0, gray2.shape[1])
        sub_img = gray2[row_start:row_end, col_start:col_end]
        found = hog.detectMultiScale(sub_img, winStride=(8, 8), padding=(30, 30), scale=1.05)[0]
        # print len(found)
        if len(found):
            # draw human detected drift box as green
            # cv2.rectangle(frame2, (col_start, row_start), (col_end, row_end), (0, 255, 0), 1)
            for ri, r in enumerate(found):
                # for qi, q in enumerate(found):
                #    if ri != qi and inside(r, q):
                #        break
                #    else:
                #        found_filtered.append(q)
                rx, ry, rw, rh = r
                # draw *ALL* human bounding as blue
                pad_w, pad_h = int(0.05*rw), int(0.05*rh)
                # cv2.rectangle(frame2, (rx+col_start+pad_w, ry+row_start+pad_h)
                #              , (rx+rw+col_start-pad_w, ry+rh+row_start-pad_h), (255, 0, 0), 2)
                # -------- initialize the correlation tracker by the *!WELL CONVINCED!* hog human bounding box ----- #
                tracker = dlib.correlation_tracker()
                tracker.start_track(gray2, dlib.rectangle(rx+col_start+pad_w, ry+row_start+pad_h,
                                                          rx+rw+col_start-pad_w, ry+rh+row_start-pad_h))
                track_list.append(tracker)
        else:
            # draw *NO* human detected drift box as red
            # l_t, t_t, r_t, b_t = convert2true_coord(col_start, row_start, col_end, row_end)
            # cv2.rectangle(frame2, (l_t, t_t), (r_t, b_t), (0, 0, 255), 1)
            cv2.rectangle(frame2, (col_start, row_start), (col_end, row_end), (0, 0, 255), 1)
        # draw frame_difference bounding box on the frame in black
        # cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 0), 1)


    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame2)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    if key == ord("p"):
        time.sleep(5)



# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()