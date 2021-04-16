#I wrote this code after finishing this course
#https://www.linkedin.com/learning/opencv-for-python-developers/
import numpy as np
import cv2

video = cv2.VideoCapture(0)

path = "haarcascade_eye.xml"
eyes_cascade = cv2.CascadeClassifier(path)

while(True):
    #get each frame from the video feed
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10,10))

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 1)
    cv2.imshow("Image", frame)

    ch = cv2.waitKey(1)
    if ch & 0xff == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
