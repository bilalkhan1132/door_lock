# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:17:09 2023

@author: WIZ TECH
"""

import cv2
from picamera.array import PiRGBArray 
from picamera import PiCamera
import numpy as np
import pickle
import time
from time import sleep
import PRi.GPIO as GPIO

relay_pin=[26]

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(relay_pin, GPIO.OUT)
#GPIO.output(relay_pin, 1)

camera= PiCamera()
camera.resolution(640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size= (640, 480))

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer.read('trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX
liste= ["Bilal", "Unknown"]

for frame in camera.capture_continuous (rawCapture, format="bgr", use_video_port= True):
    frmae= frame.array
    gray= cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMutiScale(gray, scaleFactor = 1.5, minNeighbors= 5)
    for (x,y,w,h) in faces:
        roiGray= gray[y:y+h, x:x+w]
        id_, conf= recognizer.predict (roiGray)
        
        if conf <95 :
            name= liste[id_]
            #GPIO.output (relay_pin, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText (frame, name+ str(conf), (x,y), font, 2, (0,0,255), 2, cv2.LINE_AA)
            #time.sleep (5)
            #GPIO.output (relay_pin, 1)
           
        else:
            #GPIO.output (relay_pin, 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText (frame, "Unknown", (x,y), font, 2, (0,0,255), 2, cv2.LINE_AA)
            
    
    cv2.imshow('frame', frame)
    key= cv2.waitKey(1)
    rawCapture.truncate(0)
    if key== 27:
      break

cv2.destroyAllWindows()
