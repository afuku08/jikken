import cv2

capture = cv2.VideoCapture(1)
    #capture.set(cv2.CAP_PROP_FPS, 60)
if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました")
ret, img = capture.read()
#img = img[180:300, 223:403] #'ゴー'の位置
#img = img[66:116, 105:208] #winの場合
#img = img[91:170, 109:209] #loseの場合
img = img[91:170, 433:533]

#cv2.imshow("banmen", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("lose_d_e.png", img)