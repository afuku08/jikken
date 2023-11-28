import cv2

capture = cv2.VideoCapture(1)
    #capture.set(cv2.CAP_PROP_FPS, 60)
if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました")
ret, img = capture.read()
#img = img[180:300, 223:403]
#cv2.imshow("banmen", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("banmen2.png", img)