import cv2
import numpy as np
'''
lose = cv2.imread('lose.png')
lose_hist = cv2.calcHist([lose], [2], None, [256], [0, 256])
new_lose = cv2.imread('lose1.png')
new_lose_hist = cv2.calcHist([new_lose], [2], None, [256], [0, 256])
'''
win = cv2.imread('win.png')
win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
new_win = cv2.imread('win1.png')
new_win_hist = cv2.calcHist([new_win], [2], None, [256], [0, 256])

capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

ret, img = capture.read()

count = 0
while ret:
    win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    #lose_now_hist = cv2.calcHist([img[91:170, 109:209]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(win_hist, win_now_hist, 0)
    new_comp_percent = cv2.compareHist(new_win_hist, win_now_hist, 0)
    print(str(comp_percent) + " " + str(new_comp_percent))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    ret, img = capture.read()

print("finish")