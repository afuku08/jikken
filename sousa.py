import pydirectinput as direct
import threading

direct.PAUSE = 0.005

class Sousa:
        
    def right():
        direct.press('d')
        
    def left():
        direct.press('a')
        
    def right_rotation():
        direct.press('e')
        
    def left_rotation():
        direct.press('q')
        
    def drop():
        direct.press('s')
        
    def no1(self):
        Sousa.left()
        Sousa.left()
    
    def no2(self):
        Sousa.left()   
        
    def no4(self):
        Sousa.right()
    
    def no5(self):
        Sousa.right()
        Sousa.right()
    
    def no6(self):
        Sousa.right()
        Sousa.right()
        Sousa.right()
        
    def no7(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.no1()
        
    def no8(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.no2()
    
    def no9(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
    
    def no10(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.no4()
        
    def no11(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.no5()
        
    def no12(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.no6()
    
    def no13(self):
        Sousa.right_rotation()
    
    def no14(self):
        Sousa.right_rotation()
        Sousa.right()    
        
    def no15(self):
        Sousa.right_rotation()
        Sousa.right()
        Sousa.right()

    def no16(self):
        Sousa.right_rotation()
        Sousa.left()

    def no17(self):
        Sousa.right_rotation()
        Sousa.left()
        Sousa.left()
    
    def no18(self):
        Sousa.left_rotation()
        
    def no19(self):
        Sousa.left_rotation()
        Sousa.left()
    
    def no20(self):
        Sousa.left_rotation()
        Sousa.right()
    
    def no21(self):
        Sousa.left_rotation()
        Sousa.right()
    
    def no22(self):
        Sousa.left_rotation()
        Sousa.right()

import time
sousa = Sousa()
string = "sousa.no1()"
start = time.time()
exec(string)
print(time.time() - start)