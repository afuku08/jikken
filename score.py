import cv2
img = cv2.imread("frames/frame1.jpg")

score1 = img[395:420, 100:214]
score2 = img[395:420, 424:538]

cv2.imshow("socre1", score1)
cv2.imshow("socre2", score2)
cv2.waitKey(0)
cv2.destroyAllWindows()

import pytesseract
from PIL import Image

number = pytesseract.image_to_string(score1)
print(number)