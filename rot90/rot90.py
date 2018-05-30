import numpy as np
import cv2


cap = cv2.VideoCapture('./t4.mp4')
    # vidcap = cv2.VideoCapture('myvid2.mp4')
success, image = cap.read()
# key = ''
while True:
	# img=cv2.imread("trunk1.jpg",1)
	# cv2.imshow("temp",img)
	# print('1111111111111',img)

	# cv2.waitKey(0)
	success, image = cap.read()
 
	img90=np.rot90(image)
	# print('22222222222',img90)
	cv2.imshow("rotate",img90)
	cv2.waitKey(50)
	# key  = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
