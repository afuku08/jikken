import cv2
import numpy as np

img = cv2.imread("Screenshot 2023-10-24 18-55-08.png")
img = cv2.resize(img, (640, 480))
cv2.imshow("banmen", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
hist = cv2.calcHist([img], [2], None, [256], [0, 256])
print(type(cv2.compareHist(hist, hist, 0)))


