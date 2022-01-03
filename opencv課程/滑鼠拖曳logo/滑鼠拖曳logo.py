import cv2
import numpy as np

img2 = cv2.imread('logo.png')
img2 = cv2.resize(img2, (100, 100)) 
rowsLogo,colsLogo,channelsLogo = img2.shape
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
rowsFrame, colsFrame, channelsFrame = frame.shape
mode = True


def InsertLogo(i, j):
	if (j+rowsLogo < rowsFrame and i+colsLogo<colsFrame):
		roi = frame[j:j+rowsLogo, i:i+colsLogo]
		dst_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
		dst_fg = cv2.bitwise_and(img2,img2,mask = mask)
		dst = cv2.add(dst_bg,dst_fg)
		frame[j:j+rowsLogo, i:i+colsLogo] = dst


# mouse callback function
def Get_Logo(event,x,y,flags,param):
	print(x, y)
	global frame, mode

	if event == cv2.EVENT_LBUTTONDOWN:
		mode = False

	if event == cv2.EVENT_MOUSEMOVE:
		if (mode == False):
			InsertLogo(x, y)
			cv2.imshow('frame',frame)       

	if event == cv2.EVENT_LBUTTONUP:
		mode = True

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', Get_Logo)


xLogo=0
yLogo=100

xxLogo=0
yyLogo=100
while(True):
	ret, frame = cap.read()
	frame=cv2.flip(frame,1)
	# Insert the first Logo
	if (mode == True):
		InsertLogo(xLogo, yLogo)
		InsertLogo(xxLogo, yyLogo)
		xLogo  = xLogo+10
		yyLogo = yyLogo + 10
		if (xLogo+colsLogo>colsFrame):
			xLogo = 0
		if (yyLogo+rowsLogo>rowsFrame):
			yyLogo = 0
		# Show the result
		cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()