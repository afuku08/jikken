import cv2
import collections
import numpy as np
import sousa
import time
import threading

capture = cv2.VideoCapture(1)
sousa = sousa.sousa()

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました")
ret, img = capture.read()
count = 0
arr1 = []
arr2 = []
q1 = collections.deque([], 4)
q2 = collections.deque([], 4)

count = 0
while ret:
    start = time.time()

    player1_next = img[73 : 123 , 240 : 260]
    player1_next_next = img[132 : 172 , 259 : 274]
    #player1_next = cv2.cvtColor(player1_next, cv2.COLOR_BGR2GRAY)
    #player1_next_next = cv2.cvtColor(player1_next_next, cv2.COLOR_BGR2GRAY)

    q1.append(player1_next)
    q2.append(player1_next_next)
    
    '''
    if len(q1) == 3:
        arr1.append(np.array_equal(q1[0], q1[2]))
        arr2.append(np.array_equal(q2[0], q2[2]))
    '''
    if len(q1) == 4:
        next1_0 = cv2.calcHist([q1[0]], [2], None, [256], [0, 256])
        next1_1 = cv2.calcHist([q1[1]], [2], None, [256], [0, 256])
        next1_2 = cv2.calcHist([q1[2]], [2], None, [256], [0, 256])
        next1_3 = cv2.calcHist([q1[3]], [2], None, [256], [0, 256])
        #print(str(cv2.compareHist(next1_0, next1_2, 0)))
        if cv2.compareHist(next1_0, next1_2, 0) < 0.5:
            print("tumo" + str(count))
            print()
            count += 1
            q1.clear()

        
    #sousa.drop()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    ret, img = capture.read()
    end = time.time()
    #print(end - start)

cv2.destroyAllWindows()