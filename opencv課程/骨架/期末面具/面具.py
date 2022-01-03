import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
def gray_transfer(gray):
    rows, cols = gray.shape
    for i in range(rows):
        for j in range(cols):
            if gray[i,j] == 116:
                gray[i,j] = 255
            elif gray[i,j] == 255:
                gray[i,j] = 0
    return
def InsertMask(img1, img2, x, y):
    rows, cols, channels = img2.shape
    if x + rows > img1.shape[0] or y + cols > img1.shape[1] or x < 1 or y <1:
        return img1

    roi = img1[x:x + rows, y:y + cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_transfer(img2gray)
    ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[x:x + rows, y:y + cols] = dst

    return img1
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
rowsFrame,colsFrame,channelsFrame = frame.shape

blank_image = np.zeros((len(frame),len(frame[0]),3), np.uint8)
logo = cv2.imread('fox.png')
logo = cv2.resize(logo, (280, 250))
rowsLogo,colsLogo,channelsLogo = logo.shape

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        ###############################################
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, c = image.shape
        cx = int(results.pose_landmarks.landmark[0].x * w) - 140
        cy = int(results.pose_landmarks.landmark[0].y * h) - 180
        #cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        print(cx, cy)


        InsertMask(frame, logo, cy, cx)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break