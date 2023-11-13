import cv2
import collections
import numpy as np
import sousa
import time
import threading

capture = cv2.VideoCapture(1)
#sousa = sousa.sousa()

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました")
ret, img = capture.read()
count = 0
q1 = collections.deque([], 5)
q2 = collections.deque([], 5)

count = 0
while ret:
    start = time.time()

    player1_next = img[73 : 123 , 240 : 260]
    player1_next_next = img[132 : 172 , 259 : 274]
    #player1_next = cv2.cvtColor(player1_next, cv2.COLOR_BGR2GRAY)
    #player1_next_next = cv2.cvtColor(player1_next_next, cv2.COLOR_BGR2GRAY)

    #cv2.resize(player1_next_next,(50, 20))
    q1.append(player1_next)
    q2.append(player1_next_next)
    
    if len(q1) == 5:
        #next1_0 = cv2.calcHist([q1[0]], [2], None, [256], [0, 256])
        #next1_1 = cv2.calcHist([q1[1]], [2], None, [256], [0, 256])
        #next1_3 = cv2.calcHist([q1[3]], [2], None, [256], [0, 256])
        #next1_4 = cv2.calcHist([q1[4]], [2], None, [256], [0, 256])
        #next2_0 = cv2.calcHist([q2[0]], [2], None, [256], [0, 256])
        #next2_1 = cv2.calcHist([q2[1]], [2], None, [256], [0, 256])
        #next2_3 = cv2.calcHist([q2[3]], [2], None, [256], [0, 256])
        #next2_4 = cv2.calcHist([q2[4]], [2], None, [256], [0, 256])
        #print(str(cv2.compareHist(next1_0, next1_2, 0)))
        #flag1 = cv2.compareHist(next1_0, next1_3, 0) < 0.7 and cv2.compareHist(next2_0, next2_3, 0) < 0.7
        #flag2 = cv2.compareHist(next1_1, next1_4, 0) > 0.9 and cv2.compareHist(next2_1, next2_4, 0) > 0.9
        #print(str(flag1) + " " + str(flag2))
        #print(str(cv2.compareHist(next1_0, next1_3, 0))[0:4] + " " + str(cv2.compareHist(next1_1, next1_4, 0))[0:4] + " " + str(cv2.compareHist(next2_0, next2_3, 0))[0:4] + " " + str(cv2.compareHist(next2_1, next2_4, 0))[0:4])
        
        print(str(np.count_nonzero(q1[0]==q1[3])/q1[0].size)[0:4] + " " + str(np.count_nonzero(q1[1]==q1[4])/q1[1].size)[0:4] + " " + str(np.count_nonzero(q2[0]==q2[3])/q2[0].size)[0:4] + " " + str(np.count_nonzero(q2[1]==q2[4])/q2[1].size)[0:4])
        '''
        if flag1 and flag2:
            print("tumo" + str(count))
            print()
            count += 1
            q1.clear()
            q2.clear()
        '''
        
    #sousa.drop()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    ret, img = capture.read()
    end = time.time()
    #print(end - start)

cv2.destroyAllWindows()