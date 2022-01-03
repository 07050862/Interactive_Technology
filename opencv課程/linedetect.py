import cv2
import math
import numpy as np
cap=cv2.VideoCapture(0)

while(True):
	ret, src=cap.read()
	dst = cv2.Canny(src, 50, 200)

	lines = cv2.HoughLines(dst, 1, math.pi / 180.0, 50, np.array([]), 0, 0)

	if lines is not None:
		a, b, c = lines.shape
		# for i in range(a):
		for i in range(20):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0, y0 = a * rho, b * rho
			pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
			pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
			cv2.line(src, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

	cv2.imshow("source", src)
	cv2.imshow("detected lines", dst)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.waitKey(0)
cv2.destroyAllWindows()