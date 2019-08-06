import cv2
import os
import numpy as np
import faceRecognition as fr
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time

try:
    while True:
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", help="path to the video file")
        ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
        args = vars(ap.parse_args())
        
        # if the video argument is None, then we are reading from webcam
        if args.get("video", None) is None:
            vs = VideoStream(src=0).start()
            # time.sleep(2.0)
        
        # otherwise, we are reading from a video file
        else:
            vs = cv2.VideoCapture(args["video"])
        
        # initialize the first frame in the video stream
        firstFrame = None
        count = 0

        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text
            frame = vs.read()
            frame = frame if args.get("video", None) is None else frame[1]
            text = "Unoccupied"
            
            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if frame is None:
                break
        
            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
        
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                    continue
        
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
            # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imwrite('NEW/img%d.jpg' % count, frame)
            count = count +1
            time.sleep(3)
            if(text == "Occupied"):
                cv2.imwrite("security.jpg",frame)
                
                break
        face_cascade = cv2.CascadeClassifier('New folder/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('New folder/haarcascade_eye.xml')
        img = cv2.imread('security.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        try:
            if len(faces) != 0:
                faces_detected,gray_img=fr.faceDetection(img)
                #print("faces_detected:",faces_detected)
                try:
                    if len(faces_detected) != 0:
                        print('detected' )
                        
                    else:
                        print('not detected')
                        
                except:
                    print('error')
                else:
                    print('not error')            
                
            else :
                print("ERROR")
                
        except:
            print("NO")
        else:
            print('Yes')    	

except:
    print("unable to capture")
else:
    print("OK")
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()