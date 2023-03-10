import cv2
import os
#from cv2 import videoCapture
from cv2 import waitKey
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#for each person we enter one numeric face id

face_id = input('\n Enter user id end press <return> ==> ')
print('\n (INFO) Intializing face capture. Look the camera and wait ...')

count=0

while(True):
    ret, img = cam.read()
    img = cv2.flip(img, 1) #flip the video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h) ,(255,0,0), 2)
        count+= 1
        #save the captured image into the datsets floder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count)+".jpg", gray[y: y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  #press ESC for existing video
    if k == 27:
       break
    elif count >= 90:  #we are going to take 90 face capture
     break

print("\n [INFO] Existing Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
