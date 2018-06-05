import numpy as np
import cv2

human_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_upperbody.xml')

cap = cv2.VideoCapture('../../Data/vid3_IR.avi')



while(cap.isOpened()):
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    humans = human_cascade.detectMultiScale(gray_img, 1.3, 5)

    for (x,y,w,h) in humans:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
