import numpy as np
import cv2

cap = cv2.VideoCapture("../../Data/vid3_IR.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()

fgbg.setHistory(20)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)

    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(fgmask,kernel,iterations = 1)
    cv2.imshow('erosion',erosion)

    kernel = np.ones((10,10),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    cv2.imshow('dilation',dilation)

    ret,thresh = cv2.threshold(dilation,127,255,0)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
 
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    cv2.imshow('org',frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
