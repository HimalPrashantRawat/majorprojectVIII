import cv2
import os
import numpy as np
import faceRecognition as fr
import motiondetector


face_cascade = cv2.CascadeClassifier('cascades/face.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
motion_detection = motiondetector.motion_detect()
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
	
if len(faces)!= 0:
	faces_detected,gray_img=fr.faceDetection(img)
	print("faces_detected:",faces_detected)	
	if len(faces_detected) !=0:
		print("detected" )
	else:
		print("not detected")
else:
	()

		
cv2.imshow('frame',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
