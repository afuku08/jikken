import cv2

start = cv2.imread('go.png')
go_hist = cv2.calcHist([start], [2], None, [256], [0, 256])
capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

ret, img = capture.read()
#count = 0

while ret:
    cv2.imshow('banmen', img[180:300, 223:403])
    now_hist = cv2.calcHist([img[180:300, 223:403]], [2], None, [256], [0, 256])
    print(str(cv2.compareHist(go_hist, now_hist, 0)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    ret, img = capture.read()


cv2.destroyAllWindows()