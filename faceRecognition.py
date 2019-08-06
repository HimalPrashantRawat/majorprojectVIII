import cv2
import os
import numpy as np

def faceDetection(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('cascades/myhaar.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles
    return faces,gray_img
