import numpy as np
import cv2
import matplotlib.pyplot as pl
#import os
#import time

cap = cv2.VideoCapture(0)
#cap.release()
T = 20

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	gray[gray <= T] = 0
	gray[gray > T] = 255
	#ret, thresh = cv2.threshold(gray, 127, 255, 0)
	#im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(gray, contours, -1, (0,255,0), 3) 


	if (cv2.waitKey(1) & 0xFF) == ord('q'):
		break

	if (cv2.waitKey(1) & 0xFF) == ord('z'):
		T += 10

	if (cv2.waitKey(1) & 0xFF) == ord('x'):
		T -= 10

	if T > 255:
		T = 255

	if T < 0:
		T = 0 

	# Display the resulting frame
	cv2.imshow('frame',gray)
	#print(hist)
	#time.sleep(1)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 

hist = cv2.calcHist([frame],[0],None,[256],[0,256])
#print(hist)
pl.plot(np.array(hist),'r')
pl.show()

















