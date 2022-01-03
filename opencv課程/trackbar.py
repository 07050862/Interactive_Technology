import numpy as np
import cv2

def nothing(x):
	pass
cap = cv2.VideoCapture(0)


cv2.namedWindow('BW')
cv2.createTrackbar('Trackbar1', 'BW', 100, 255, nothing)

while(True):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Our operations on the frame come here
    thrs1 = cv2.getTrackbarPos('Trackbar1', 'BW')
    ret, balck_and_white = cv2.threshold(gray, thrs1, 255, cv2.THRESH_BINARY)
    cv2.imshow('frame', frame)
    cv2.imshow('BW', balck_and_white)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()