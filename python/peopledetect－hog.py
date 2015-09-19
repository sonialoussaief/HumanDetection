#!/usr/bin/env python

import numpy as np
from cv2 import cv
import cv2
# from video import create_capture

help_message = '''
USAGE: peopledetect.py <image_names> ...

Press any key to continue, ESC to stop.
'''

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.15*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x, y

if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print help_message

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


    cv2.namedWindow('img') # Can be resized
    cv2.namedWindow('dst')
    cv2.setMouseCallback('img',mouse_callback)

    #cap = create_capture(1)
    cap = cv2.VideoCapture('/Users/Alex/Desktop/HumanDetection/video/Video5.mov')
    ret, img = cap.read()
    print cap.get(cv.CV_CAP_PROP_FPS)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    f = open('/Users/Alex/Desktop/HumanDetection/position_test.txt','w')


    i = 0
    buffer = []
    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        #img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rows,cols = img.shape

        print rows,cols
        pts1 = np.float32([[470,387],[626, 382],[693,416],[472,430]])

#       pts2 = np.float32([[400,400],[600,400],[600,600],[400,600]])

        pts2 = np.float32([[765,0],[460,0],[460,305],[765,305]])

        pts_real = np.float32([[7.65,0],[4.6,0],[4.6,3.05],[7.65,3.05]])


#        pts1 = np.float32([[100,100],[200,100],[200,200],[100,200]])
#        pts2 = np.float32([[0,0],[100,0],[100,100],[0,100]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        M_real = cv2.getPerspectiveTransform(pts1,pts_real)


        found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(0,0), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
                else:
                    found_filtered.append(q)
#        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print '%d (%d) found' % (len(found_filtered), len(found))
	
	if len(found) == 2 & len(found_filtered) == 3:
	    print found
	    print r

        for x, y, w, h in found:
            pad_w, pad_h = int(0.15*w), int(0.2*h)
            src_point = np.float32([[[x+w-pad_w/2, y+h-pad_h]]])
            dst_point = cv2.perspectiveTransform(src_point, M_real)
            buffer.append(dst_point)
	    
   	    #print x, y, w, h

        i = i + 1
        if i % 30 == 0:
            pp = np.average(buffer,axis=0)
            print >>f, "%f,%f" % (pp[0][0][0], pp[0][0][1])
            f.flush()
            buffer = []

        #original
        cv2.imshow('img', img)
        #transfered
        dst = cv2.warpPerspective(img,M,(cols,rows))
        cv2.imshow('dst', dst)


        if 0xFF & cv2.waitKey(5) == 27:
            break

    cv2.destroyAllWindows()
