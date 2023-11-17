import cv2

img = cv2.imread('banmen1.png')
th, img_th = cv2.threshold(img, 111, 255, cv2.THRESH_BINARY)
cv2.imshow('imt', img_th)
cv2.waitKey()
cv2.destroyAllWindows()