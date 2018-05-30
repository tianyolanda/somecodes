import cv2 
import os

m = '2018_'
n = 1
for i in range(121,144):

	vc = cv2.VideoCapture('/home/ogai1234/Desktop/video/WeChatSight'+str(i)+'.mp4')
	c = 1
	t = 10
	rval = vc.isOpened()
	while rval:
		rval, frame = vc.read()
		if(c%t == 0):
			cv2.imwrite('/home/ogai1234/Desktop/piccc/' + m + '{0:06d}'.format(n/10) + '.jpg',frame)
		c = c + 1
		n = n + 1
	vc.release()
