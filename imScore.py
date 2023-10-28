import pytesseract
from PIL import Image, ImageOps
import cv2
import pyocr
import pyocr.builders

'''
im = Image.open('frames/frame1.jpg')
im_crop = im.crop((100,395, 214,420))
im_crop = im_crop.convert('L')
im_crop = ImageOps.invert(im_crop)
im_crop.show()
'''

image = cv2.imread('frames/frame1.jpg', 0)
image = image[395:420, 100:214]
ret2, img_otsu = cv2.threshold(image,240,255,cv2.THRESH_BINARY)
img_otsu = cv2.bitwise_not(img_otsu)
#cv2.imshow('img_otsu', img_otsu)
#cv2.waitKey()
#cv2.destroyAllWindows()

IMG_OTSU = Image.fromarray(img_otsu)
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
tools = pyocr.get_available_tools()
tool = tools[0]

builder = pyocr.builders.TextBuilder(tesseract_layout=6)

builder = pyocr.builders.TextBuilder()
start = time.time()
result = tool.image_to_string(IMG_OTSU, lang="jpn", builder=builder)
end = time.time()
print(result)
print(end-start)