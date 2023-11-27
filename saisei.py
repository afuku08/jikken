import cv2
import numpy as np
from collections import deque
import collections
import threading

go = cv2.imread('go1.png')
go_hist = cv2.calcHist([go], [2], None, [256], [0, 256])
def start_judge(img):
    #cv2.imshow('banmen', img[180:300, 223:403])
    now_hist = cv2.calcHist([img[180:300, 223:403]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(go_hist, now_hist, 0)
    if comp_percent > 0.95:
        return True
    else:
        return False

win = cv2.imread('win1.png')
win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
def win_judge(img):
    win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(win_hist, win_now_hist, 0)
    if comp_percent >= 0.95:
        return True
    else:
        return False

lose = cv2.imread('lose1.png')
lose_hist = cv2.calcHist([lose], [2], None, [256], [0, 256])
def lose_judge(img):
    lose_now_hist = cv2.calcHist([img[91:170, 109:209]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(lose_hist, lose_now_hist, 0)
    if comp_percent >= 0.95:
        return True
    else:
        return False

import time
def main():
    capture = cv2.VideoCapture(1)

    if (capture.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました") 
    count = 0
    ret, img = capture.read()
    #for i in range(50):
    while True:
        win_flag = False
        lose_flag = False
        #arr1 = []
        #arr2 = []
        q1 = collections.deque([], 4)
        q2 = collections.deque([], 4)
        if start_judge(img):
            while True:
                start = time.time()
                win_flag = win_judge(img)
                lose_flag = lose_judge(img)
                if win_flag or lose_flag:
                    count += 1
                    break
                player1_next = img[73 : 123 , 240 : 260]
                player1_next_next = img[132 : 172 , 259 : 274]
                player1_next = cv2.cvtColor(player1_next, cv2.COLOR_BGR2GRAY)
                player1_next_next = cv2.cvtColor(player1_next_next, cv2.COLOR_BGR2GRAY)
                q1.append(player1_next)
                q2.append(player1_next_next)
                if len(q1) == 4:
                    flag1 = (np.array_equal(q1[0], q1[1]) == 0) and (np.array_equal(q2[0], q2[1]) == 0)
                    flag2 = (np.array_equal(q1[2], q1[3]) == 1) and (np.array_equal(q2[2], q2[3]) == 1)

                    if flag1 and flag2:
                        print('tumo')
                    
                        q1.clear()
                        q2.clear()
                print(1)
                ret, img = capture.read()
                end = time.time()
                print(end - start)
        else:
            ret, img = capture.read()
            continue
    
        print(0)
        ret, img = capture.read()

    
if __name__ == "__main__":
    main()