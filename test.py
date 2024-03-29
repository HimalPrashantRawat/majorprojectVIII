import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/myhaar.xml')
eye_cascade = cv2.CascadeClassifier('cascades/myhaar_eye.xml')
img = cv2.imread('frame102.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
	img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
	if len(faces)!=0:
		print("detected")
	else:
		print("not detected")

cv2.imshow("img",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
