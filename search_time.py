import cv2
import numpy as np
import time
import collections
import pyocr
import pyocr.builders
import threading
from PIL import Image, ImageOps

capture = cv2.VideoCapture(1)

if (capture.isOpened()== False):  
    print("ビデオファイルを開くとエラーが発生しました") 

tools = pyocr.get_available_tools()
tool = tools[0]
#builder = pyocr.builders.TextBuilder(tesseract_layout=6)
builder = pyocr.builders.DigitBuilder(tesseract_layout=6)
#builder = pyocr.builders.TextBuilder()
results = ['', '']

def get_score1(score1):
    score = tool.image_to_string(score1, builder=builder)
    if score.isdecimal():
        results[0] = int(score)

def get_score2(score2):
    score = tool.image_to_string(score2, builder=builder)
    if score.isdecimal():
        results[1] = int(score)

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
    #print(results[0])
    #print(results[1])
    
win = cv2.imread('win.png')
win_hist = cv2.calcHist([win], [2], None, [256], [0, 256])
def win_judge(img):
    win_now_hist = cv2.calcHist([img[66:116, 105:208]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(win_hist, win_now_hist, 0)
    if comp_percent >= 0.6:
        return True
    else:
        return False

lose = cv2.imread('lose.png')
lose_hist = cv2.calcHist([lose], [2], None, [256], [0, 256])
def lose_judge(img):
    lose_now_hist = cv2.calcHist([img[91:170, 109:209]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(lose_hist, lose_now_hist, 0)
    if comp_percent >= 0.2:
        return True
    else:
        return False


ret, img = capture.read()
count = 0
count_time = 1
q1 = collections.deque([], 5)
q2 = collections.deque([], 5)
while ret:
    start = time.time()
    count_time += 1
    if count_time % 30 == 0:
        get_score(img)
        count_time = 1
    win_flag = win_judge(img)
    lose_flag = lose_judge(img)
    if win_flag or lose_flag:
        count += 1
        break
    player1_next = img[73 : 123 , 240 : 260]
    player1_next_next = img[132 : 172 , 259 : 274]
    player1_next = cv2.cvtColor(player1_next, cv2.COLOR_BGR2GRAY)
    player1_next_next = cv2.cvtColor(player1_next_next, cv2.COLOR_BGR2GRAY)
    q1.append(player1_next)
    q2.append(player1_next_next)
    if len(q1) == 5:
        flag1 = (np.array_equal(q1[0], q1[3]) == 0) and (np.array_equal(q2[0], q2[3]) == 0)
        flag2 = (np.array_equal(q1[1], q1[4]) == 1) and (np.array_equal(q2[1], q2[4]) == 1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    print(time.time() - start)
    ret, img = capture.read()

print("finish")