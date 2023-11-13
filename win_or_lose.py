import cv2
import numpy as np
import time


win = cv2.imread('win.png')
win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
def win_judge(img):
    win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(win_hist, win_now_hist, 0)
    if comp_percent >= 0.6:
        return True
    else:
        return False

lose = cv2.imread('lose.png')
lose_hist = cv2.calcHist([lose], [2], None, [256], [0, 256])
def lose_judge(img):
    lose_now_hist = cv2.calcHist([img[91:170, 109:209]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(lose_hist, lose_now_hist, 0)
    if comp_percent >= 0.2:
        return True
    else:
        return False


capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

ret, img = capture.read()

count = 0
while ret:
    start = time.time()
    win_flag = win_judge(img)
    lose_flag = lose_judge(img)
    print(str(win_flag) + " " + str(lose_flag))
    if win_flag or lose_flag:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    print(time.time() - start)
    ret, img = capture.read()

print("finish")