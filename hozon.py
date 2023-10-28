import cv2

img = cv2.imread("Screenshot 2023-10-24 18-55-08.png")
img = cv2.resize(img, (640, 480))
img = img[91:170, 109:209]
cv2.imshow("banmen", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("lose.png", img)