import pytesseract
from PIL import Image, ImageOps
import cv2
import pyocr
import pyocr.builders
import threading

'''
im = Image.open('frames/frame1.jpg')
im_crop = im.crop((100,395, 214,420))
im_crop = im_crop.convert('L')
im_crop = ImageOps.invert(im_crop)
im_crop.show()
'''

tools = pyocr.get_available_tools()
tool = tools[0]
#builder = pyocr.builders.TextBuilder(tesseract_layout=6)
builder = pyocr.builders.DigitBuilder(tesseract_layout=6)
#builder = pyocr.builders.TextBuilder()
results = ['', '']

def get_score1(score1):
    results[0] = tool.image_to_string(SCORE1, builder=builder)

def get_score2(score2):
    results[1] = tool.image_to_string(SCORE2, builder=builder)

def get_score(image):
    ret2, image = cv2.threshold(image,240,255,cv2.THRESH_BINARY)
    image = cv2.bitwise_not(image)
    score1 = image[395:420, 100:214]
    score2 = image[395:420, 424:538]

    SCORE1 = Image.fromarray(score1)
    SCORE2 = Image.fromarray(score2)

    thread1 = threading.Thread(target=get_score1, args=(SCORE1,))
    thread2 = threading.Thread(target=get_score2, args=(SCORE2,))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    #result1 = tool.image_to_string(SCORE1, builder=builder)
    #result2 = tool.image_to_string(SCORE2, builder=builder)
    print(results[0])
    print(results[1])

image = cv2.imread('frames/frame1.jpg', 0)
ret2, image = cv2.threshold(image,240,255,cv2.THRESH_BINARY)
image = cv2.bitwise_not(image)
score1 = image[395:420, 100:214]
score2 = image[395:420, 424:538]
#ret2, img_otsu = cv2.threshold(image,240,255,cv2.THRESH_BINARY)
#img_otsu = cv2.bitwise_not(img_otsu)
#cv2.imshow('img_otsu', img_otsu)
#cv2.waitKey()
#cv2.destroyAllWindows()

SCORE1 = Image.fromarray(score1)
SCORE2 = Image.fromarray(score2)
#IMG_OTSU = Image.fromarray(img_otsu)
#IMG_OTSU.show()
#num = pytesseract.image_to_string(IMG_OTSU)
#print(num)
#img = Image.open('68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3133303138312f39373062383432632d613364642d353237352d376364332d3463636130373865346235622e706e67.png')

'''
import time
start = time.time()
number = pytesseract.image_to_string(im_crop)
end = time.time()
print(end - start)
print(type(number))
print(number)
'''

import time
'''
tools = pyocr.get_available_tools()
tool = tools[0]

builder = pyocr.builders.TextBuilder(tesseract_layout=6)

builder = pyocr.builders.TextBuilder()
start = time.time()
result1 = tool.image_to_string(SCORE1, builder=builder)
result2 = tool.image_to_string(SCORE2, builder=builder)
end = time.time()
print(result1)
print(result2)
print(end-start)
'''

start = time.time()
get_score(image)
print(time.time() - start)
print(type(results[0]))