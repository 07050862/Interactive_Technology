import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def nothing(x):
	pass

finishedColorSelection = False
TH1=100
TH2=140

def mouseCallback(event,x,y,flags,param):
    global finishedColorSelection, TH1, TH2
    if event == cv2.EVENT_LBUTTONDOWN:
        finishedColorSelection = True
        print(TH1, TH2)


cv2.namedWindow('mask')
cv2.createTrackbar('TH1', 'mask', 100, 255, nothing)
cv2.createTrackbar('TH2', 'mask', 140, 255, nothing)

cv2.setMouseCallback('mask', mouseCallback)

# Preprocess for Color Selection
while(1):
    #global finishedColorSelection
    # Take each frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define Hue range of selected color in HSV
    TH1 = cv2.getTrackbarPos('TH1', 'mask')
    TH2 = cv2.getTrackbarPos('TH2', 'mask')
    lower_Hue = np.array([TH1, 50, 50], dtype=np.uint8)
    upper_Hue = np.array([TH2,255,255], dtype=np.uint8)
    # Threshold the HSV image to get only selected colors
    mask = cv2.inRange(hsv, lower_Hue, upper_Hue)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27 or finishedColorSelection == True:
        break