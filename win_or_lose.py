import cv2
import numpy as np
import time

channel_b = 0
channel_g = 1

go = cv2.imread('go_d.png')
go_hist_b = cv2.calcHist([go], [channel_b], None, [256], [0, 256])
go_hist_g = cv2.calcHist([go], [channel_g], None, [256], [0, 256])
def start_judge(img):
    #cv2.imshow('banmen', img[180:300, 223:403])
    now_hist_b = cv2.calcHist([img[180:300, 223:403]], [channel_b], None, [256], [0, 256])
    now_hist_g = cv2.calcHist([img[180:300, 223:403]], [channel_g], None, [256], [0, 256])

    comp_percent_b = cv2.compareHist(go_hist_b, now_hist_b, 0)
    comp_percent_g = cv2.compareHist(go_hist_g, now_hist_g, 0)
    #print(comp_percent)
    if comp_percent > 0.95:
        return True
    else:
        return False

#相手の負けで勝ちを判定する
win = cv2.imread('lose_d_e.png')
win_hist_b = cv2.calcHist([win], [channel_b], None, [256], [0, 256])
win_hist_g = cv2.calcHist([win], [channel_g], None, [256], [0, 256])
def win_judge(img):
    win_now_hist_b = cv2.calcHist([img[91:170, 433:533]], [channel_b], None, [256], [0, 256])
    win_now_hist_g = cv2.calcHist([img[91:170, 433:533]], [channel_g], None, [256], [0, 256])
    comp_percent_b = cv2.compareHist(win_hist_b, win_now_hist_b, 0)
    comp_percent_g = cv2.compareHist(win_hist_g, win_now_hist_g, 0)
    comp_percent = (comp_percent_b + comp_percent_g) / 2
    print(str(comp_percent))
    if comp_percent >= 0.72:
        return True
    else:
        return False

lose = cv2.imread('lose_d.png')
lose_hist_b = cv2.calcHist([lose], [channel_b], None, [256], [0, 256])
lose_hist_g = cv2.calcHist([lose], [channel_g], None, [256], [0, 256])
def lose_judge(img):
    lose_now_hist_b = cv2.calcHist([img[91:170, 109:209]], [channel_b], None, [256], [0, 256])
    lose_now_hist_g = cv2.calcHist([img[91:170, 109:209]], [channel_g], None, [256], [0, 256])
    comp_percent_b = cv2.compareHist(lose_hist_b, lose_now_hist_b, 0)
    comp_percent_g = cv2.compareHist(lose_hist_g, lose_now_hist_g, 0)
    comp_percent = (comp_percent_b + comp_percent_g) / 2
    #print(str(comp_percent))
    if comp_percent >= 0.72:
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
    if win_flag or lose_flag:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    print(time.time() - start)
    ret, img = capture.read()

print("finish")