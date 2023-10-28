# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:53:01 2023

@author: candy
"""

import cv2
import time

win = cv2.imread('win.png')
win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
def win_judge(img):
    win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(win_hist, win_now_hist, 0)
    if comp_percent > 0.6:
        return True
    else:
        return False

start = time.time()
img = cv2.imread('banmen1.png')
flag = win_judge(img)
end = time.time()
print(end - start)

