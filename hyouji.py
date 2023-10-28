# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:50:17 2023

@author: candy
"""

import cv2
import numpy as np

img = cv2.imread("Screenshot 2023-10-24 18-55-08.png")
def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
img = cv2.resize(img, (640, 480))
cv2.imshow("banmen", img)
cv2.setMouseCallback('banmen', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()