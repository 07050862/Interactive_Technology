import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame1=cv2.flip(frame,0)
    frame2 = cv2.flip(frame, 1)
    frame3 = cv2.flip(frame1, 1)

    frame = cv2.resize(frame, (320, 240))
    frame1 = cv2.resize(frame1, (320, 240))
    frame2 = cv2.resize(frame2, (320, 240))
    frame3 = cv2.resize(frame3, (320, 240))
    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame1',frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('frame3', frame3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()