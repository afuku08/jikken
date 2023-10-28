# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:06:46 2023

@author: candy
"""
import pydirectinput as direct
import threading

class sousa:
        
    def right():
        direct.press('d')
        
    def left():
        direct.press('a')
        
    def right_rotation():
        direct.press('e')
        
    def left_rotation():
        direct.press('q')
        
    def drop(self):
        direct.press('s')
        
    def no1(self):
        thread1 = threading.Thread(target=sousa.left())
        thread2 = threading.Thread(target=sousa.left())
        thread1.start()
        thread2.start()
    
    def no2():
        sousa.left()   
        
    def no4():
        sousa.right()
    
    def no5():
        sousa.right()
        sousa.right()
    
    def no6():
        sousa.right()
        sousa.right()
        sousa.right()
        
    def no7():
        sousa.right_rotation()
        sousa.right_rotation()
        sousa.no1()
        
    def no8():
        sousa.right_rotation()
        sousa.right_rotation()
        sousa.no2()
    
    def no9():
        sousa.right_rotation()
        sousa.right_rotation()
    
    def no10():
        sousa.right_rotation()
        sousa.right_rotation()
        sousa.no4()
        
    def no11():
        sousa.right_rotation()
        sousa.right_rotation()
        sousa.no5()
        
    def no12():
        sousa.right_rotation()
        sousa.right_rotation()
        sousa.no6()
        
        
        