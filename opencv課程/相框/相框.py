import cv2
import numpy as np
cap=cv2.VideoCapture(0)
img=cv2.imread('frame.jpg')
img=cv2.resize(img,(640,480))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('mask', mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv',mask_inv)
while(True):
	ret, frame=cap.read()
	frame=cv2.resize(frame,(640,480))
	frame=cv2.flip(frame,1)
	roi=frame[:]
	roi=cv2.bitwise_and(roi,roi,mask = mask_inv)
	img=cv2.bitwise_and(img,img,mask = mask)
	dst = cv2.add(roi,img)
	cv2.imshow('dst', dst)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.waitKey(0)
cv2.destroyAllWindows()