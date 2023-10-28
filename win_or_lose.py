import cv2
import numpy as np

win = cv2.imread('win.png')
#win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
lose = cv2.imread('lose.png')
#lose_hist = cv2.calcHist([lose], [2], None, [256], [0, 256])
capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

ret, img = capture.read()
#count = 0

win_max = 0
lose_max = 0
count = 0
while ret:
    #cv2.imshow('banmen', img)
    #cv2.imshow('lose', img[91:170, 109:209])
    #win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    #lose_now_hist = cv2.calcHist([img[91:170, 109:209]], [2], None, [256], [0, 256])
    #print(str(cv2.compareHist(win_hist, win_now_hist, 0))[:4] + " " + str(cv2.compareHist(lose_hist, lose_now_hist, 0))[:4])
    win_now = img[66:116, 105:208]
    lose_now = img[91:170, 109:209]
    win_latio = np.count_nonzero(win == win_now) / win_now.size
    lose_latio = np.count_nonzero(lose == lose_now) / lose_now.size
    print(str(np.count_nonzero(win == win_now) / win_now.size) + " " + str(np.count_nonzero(lose == lose_now) / lose_now.size))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    win_max = max(win_max, win_latio)
    lose_max = max(lose_max, lose_latio)
    if count > 1000:
        break;
    count += 1
    ret, img = capture.read()

print(str(win_max) + " " + str(lose_max))
cv2.destroyAllWindows()