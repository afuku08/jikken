# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:37:53 2023

@author: candy
"""

import sousa
import time
import pydirectinput as direct

sousa_class = sousa.sousa()
for i in range(1000):
    start = time.time()
    direct.keyDown('d')
    #time.sleep(0.10)
    direct.keyUp('d')
    end = time.time()
    print(end - start)
    