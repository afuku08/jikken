# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:41:46 2023

@author: candy
"""

import cv2
import collections
import numpy as np

capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

print(capture.get(cv2.CAP_PROP_FPS))
ret, img = capture.read()
count = 0
arr1 = []
arr2 = []
q1 = collections.deque([], 3)
q2 = collections.deque([], 3)

while ret:

    player1_next = img[73 : 123 , 240 : 260]
    player1_next_next = img[132 : 172 , 259 : 274]
    
    cv2.imshow('p1_next', player1_next)
    cv2.imshow('p1_next2', player1_next_next)
    
    q1.append(player1_next)
    q2.append(player1_next_next)
    
    if len(q1) == 3:
        arr1.append(np.array_equal(q1[0], q1[2]))
        arr2.append(np.array_equal(q2[0], q2[2]))
        
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    count += 1
    print(count)
    ret, img = capture.read()
    
    if(count == 2000):
        break

cv2.destroyAllWindows()

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(arr1,color='blue')
plt.plot(arr2, color='red')
plt.show()
    