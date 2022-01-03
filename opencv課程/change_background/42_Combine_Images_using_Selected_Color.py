#利用41 先找閥值，再改18、19行
import sys
import cv2
import numpy as np


def saveFile(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.imwrite('picture.jpg', frame)
        cv2.imwrite('picture1.jpg', frame1)
        cv2.imwrite('picture2.jpg', frame2)
        cv2.imwrite('picture3.jpg', frame3)
        cv2.imwrite('picture4.jpg', frame4)
        print('SAVE FILE')


# define range of the selected color in HSV
lower_Hue = np.array([20, 50, 50], dtype=np.uint8)
upper_Hue = np.array([51, 255, 255], dtype=np.uint8)

cv2.namedWindow('frame')

cap = cv2.VideoCapture(0)

background = cv2.imread('machu.jpeg')
background = cv2.resize(background, (640, 480))
background1 = cv2.imread('london.jpg')
background1 = cv2.resize(background1, (640, 480))
background2 = cv2.imread('background.jpg')
background2 = cv2.resize(background2, (640, 480))
background3 = cv2.imread('sea.jpg')
background3 = cv2.resize(background3, (640, 480))
background4 = cv2.imread('japan.jpg')
background4 = cv2.resize(background4, (640, 480))

cv2.setMouseCallback('frame', saveFile)

while (1):

    # Take each frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_Hue, upper_Hue)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask_inv = cv2.bitwise_not(mask)

    # Bitwise-AND mask and original image
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    bg1 = cv2.bitwise_and(background1, background1, mask=mask)
    bg2 = cv2.bitwise_and(background2, background2, mask=mask)
    bg3 = cv2.bitwise_and(background3, background3, mask=mask)
    bg4 = cv2.bitwise_and(background4, background4, mask=mask)
    frame = cv2.add(fg, bg)
    frame1 = cv2.add(fg, bg1)
    frame2 = cv2.add(fg, bg2)
    frame3 = cv2.add(fg, bg3)
    frame4 = cv2.add(fg, bg4)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
sys.exit()