import numpy as np
import cv2

img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = "haarcascade_eye.xml" #the pretrained with all the nodes and weights values

eyes_cascade = cv2.CascadeClassifier(path)

eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(10,10))
print(len(eyes))

for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
