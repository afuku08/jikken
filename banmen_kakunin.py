# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 17:22:25 2023

@author: cand"""

import cv2


img = cv2.imread('banmen/banmen7.jpg')
img = cv2.resize(img, (640, 480))

#cv2.imshow('banmen', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

H = 324
W = 132
h_start = 70
player1_field = img[h_start : h_start + H, 91 : 91 + W]
player2_field = img[h_start : h_start + H, 415 : 415 + W]

h_unit = H // 12
w_unit = W // 6
count = 1

for h in range(0, H, h_unit):
    for w in range(0, W, w_unit):
        grid1 = player1_field[h : h + h_unit, w : w + w_unit]
        cv2.imwrite("./puyos/puyo_tra_1_%d.jpg" % count, grid1)
        #cv2.imshow('player1' , grid)
        #cv2.waitKey(0)
        grid2 = player2_field[h : h + h_unit, w : w + w_unit]
        cv2.imwrite("./puyos/puyo_tra_2_%d.jpg" % count, grid2)
        count += 1
    #cv2.destroyAllWindows()