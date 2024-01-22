import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
from tensorflow.keras.layers import Input
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import concatenate,add
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape,Permute# モジュールのインポート
from tensorflow.keras.layers import Conv2D,Convolution2D, MaxPooling2D,Cropping2D,Conv2DTranspose# CNN層、Pooling層のインポート
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
import numpy as np
import cv2
import random
import collections
import pyocr
import pyocr.builders
import threading
from PIL import Image, ImageOps
import pydirectinput as direct
import matplotlib.pyplot as plt

puyo_cont = []

def get_field_info(img):
    H = 324
    W = 132
    h_start = 70
    player1_field = img[h_start : h_start + H, 90 : 90 + W]
    player2_field = img[h_start : h_start + H, 415 : 415 + W]
    fields = []
    for field in [player1_field, player2_field]:
        init_field = np.zeros((12, 6), dtype=np.uint8)

        h_unit = H // 12
        w_unit = W // 6    
        
        for h in range(0, H, h_unit):
            for w in range(0, W, w_unit):
                grid = field[h : h + h_unit, w : w + w_unit]
                puyo = classifier.predict(grid, template_type="field")
                this_puyo = -1
                #からの場合
                if puyo == 6:
                    this_puyo = 0
                #お邪魔ぷよの場合
                elif puyo == 5:
                    this_puyo = 5
                #通常ぷよの場合
                else:
                    result = puyo not in puyo_cont
                    if len(puyo_cont) == 4:
                        if result:
                            this_puyo = 0
                        else:
                            this_puyo = puyo_cont.index(puyo) + 1
                    else:
                        if result:
                            puyo_cont.append(puyo)
                        
                        this_puyo = puyo_cont.index(puyo) + 1
                    
                init_field[h // h_unit, w // w_unit] = this_puyo
                
        init_field = field_edit(init_field)
        for i in range(11):
            for j in range(6):
                if init_field[i+1][j] == 0:
                    init_field[i][j] = 0

        fields.append(init_field)
    return fields
    
def get_next_puyo_info(img):
    player1_next = img[73 : 123 , 240 : 260]
    h, w, c = player1_next.shape
    player1_next = player1_next[: h // 2], player1_next[h // 2 :]
    player1_next = [classifier.predict(i, template_type="p1") for i in player1_next]
    for i in range(2):
        result = player1_next[i] not in puyo_cont
        if len(puyo_cont) == 4:
            if result:
                player1_next[i] = 0
            else:
                player1_next[i] = puyo_cont.index(player1_next[i])
        else:
            if result:
                puyo_cont.append(player1_next[i])
                        
            player1_next[i] = puyo_cont.index(player1_next[i])

    player1_next_next = img[132 : 172 , 259 : 274]
    h, w, c = player1_next_next.shape
    player1_next_next = player1_next_next[: h // 2], player1_next_next[h // 2 :]
    player1_next_next = [
        classifier.predict(i, template_type="p1") for i in player1_next_next
    ]
    for i in range(2):
        result = player1_next_next[i] not in puyo_cont
        if len(puyo_cont) == 4:
            if result:
                player1_next_next[i] = 0
            else:
                player1_next_next[i] = puyo_cont.index(player1_next_next[i])
        else:
            if result:
                puyo_cont.append(player1_next_next[i])
                        
            player1_next_next[i] = puyo_cont.index(player1_next_next[i])

    player1_nexts = [player1_next, player1_next_next]
    return player1_nexts

class puyo_classifier(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._field_template[name] = img

        self._p1_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p1/p1_{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p1_template[name] = img

        self._p2_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p2/p2_{name}.jpg"), (20, 20))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p2_template[name] = img

    def predict(self, img, template_type="field"):
        img = cv2.resize(img, (20, 20))
        differences = []

        channel_b = 0
        channel_g = 1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hist_b = cv2.calcHist([img], [channel_b], None, [256], [0, 256])
        img_hist_g = cv2.calcHist([img], [channel_g], None, [256], [0, 256])

        for name in self._puyo_types[:-2]:
            if template_type == "field":
                template_img = self._field_template[name]
            elif template_type == "p1":
                template_img = self._p1_template[name]
            elif template_type == "p2":
                template_img = self._p2_template[name]
            template_img_hist_b = cv2.calcHist(
                [template_img], [channel_b], None, [256], [0, 256]
            )
            template_img_hist_g = cv2.calcHist(
                [template_img], [channel_g], None, [256], [0, 256]
            )

            diff = cv2.compareHist(template_img_hist_b, img_hist_b, 0)
            diff += cv2.compareHist(template_img_hist_g, img_hist_g, 0)
            diff /= 2
            differences.append(diff)

        if template_type == "field":
            channel = 0
            img_hist_b = cv2.calcHist([img], [channel_b], None, [256], [0, 256])
            img_hist_g = cv2.calcHist([img], [channel_g], None, [256], [0, 256])

            for name in ["ojama", "back"]:
                satu = np.mean(img[:, :, 1])
                if satu < 5 and name == "back":
                    diff = 1.00
                else:
                    template_img = self._field_template[name]
                    template_img_hist_b = cv2.calcHist(
                        [template_img], [channel_b], None, [256], [0, 256]
                    )
                    template_img_hist_g = cv2.calcHist(
                        [template_img], [channel_g], None, [256], [0, 256]
                    )
                    diff = cv2.compareHist(template_img_hist_b, img_hist_b, 0)
                    diff += cv2.compareHist(template_img_hist_g, img_hist_g, 0)
                    diff /= 2

                differences.append(diff)
        puyo_type = differences.index(max(differences))
        return puyo_type
    
def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field


channel_b = 0
channel_g = 1

go = cv2.imread('go_d.png')
go_hist_b = cv2.calcHist([go], [channel_b], None, [256], [0, 256])
go_hist_g = cv2.calcHist([go], [channel_g], None, [256], [0, 256])
def start_judge(img):
    #cv2.imshow('banmen', img[180:300, 223:403])
    now_hist_b = cv2.calcHist([img[180:300, 223:403]], [channel_b], None, [256], [0, 256])
    now_hist_g = cv2.calcHist([img[180:300, 223:403]], [channel_g], None, [256], [0, 256])

    comp_percent_b = cv2.compareHist(go_hist_b, now_hist_b, 0)
    comp_percent_g = cv2.compareHist(go_hist_g, now_hist_g, 0)
    comp_percent = (comp_percent_b + comp_percent_g) / 2
    #print(comp_percent)
    if comp_percent > 0.9:
        return True
    else:
        return False

#相手の負けで勝ちを判定する
win = cv2.imread('lose_d_e.png')
win_hist_b = cv2.calcHist([win], [channel_b], None, [256], [0, 256])
win_hist_g = cv2.calcHist([win], [channel_g], None, [256], [0, 256])
def win_judge(img):
    win_now_hist_b = cv2.calcHist([img[91:170, 433:533]], [channel_b], None, [256], [0, 256])
    win_now_hist_g = cv2.calcHist([img[91:170, 433:533]], [channel_g], None, [256], [0, 256])
    comp_percent_b = cv2.compareHist(win_hist_b, win_now_hist_b, 0)
    comp_percent_g = cv2.compareHist(win_hist_g, win_now_hist_g, 0)
    comp_percent = (comp_percent_b + comp_percent_g) / 2
    #print(str(comp_percent))
    if comp_percent >= 0.72:
        return True
    else:
        return False

lose = cv2.imread('lose_d.png')
lose_hist_b = cv2.calcHist([lose], [channel_b], None, [256], [0, 256])
lose_hist_g = cv2.calcHist([lose], [channel_g], None, [256], [0, 256])
def lose_judge(img):
    lose_now_hist_b = cv2.calcHist([img[91:170, 109:209]], [channel_b], None, [256], [0, 256])
    lose_now_hist_g = cv2.calcHist([img[91:170, 109:209]], [channel_g], None, [256], [0, 256])
    comp_percent_b = cv2.compareHist(lose_hist_b, lose_now_hist_b, 0)
    comp_percent_g = cv2.compareHist(lose_hist_g, lose_now_hist_g, 0)
    comp_percent = (comp_percent_b + comp_percent_g) / 2
    #print(str(comp_percent))
    if comp_percent >= 0.72:
        return True
    else:
        return False


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95
        self.lr = 0.0002
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 22

        #セーブしたモデルの使用
        self.qnet = load_model('./500_model.h5')

    def get_action(self, state):
        qs = self.qnet.predict(state)
        return np.argmax(qs[0])
        
    
def map2batch(gameMap,batch_size = 1):
    return gameMap.reshape((batch_size,12,6,6))

direct.PAUSE = 0.02

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

    def no3(self):
        return
        
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
        Sousa.left()
        Sousa.left()
        
    def no8(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.left()
    
    def no9(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
    
    def no10(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.right()
        
    def no11(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.right()
        Sousa.right()
        
    def no12(self):
        Sousa.right_rotation()
        Sousa.right_rotation()
        Sousa.right()
        Sousa.right()
        Sousa.right()
    
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

sousa = Sousa()
def try_action(action):
    if action == 1:
        sousa.no1()
    elif action == 2:
        sousa.no2()
    elif action == 3:
        sousa.no3()
    elif action == 4:
        sousa.no4()
    elif action == 5:
        sousa.no5()
    elif action == 6:
        sousa.no6()
    elif action == 7:
        sousa.no7()
    elif action == 8:
        sousa.no8()
    elif action == 9:
        sousa.no9()
    elif action == 10:
        sousa.no10()
    elif action == 11:
        sousa.no11()
    elif action == 12:
        sousa.no12()
    elif action == 13:
        sousa.no13()
    elif action == 14:
        sousa.no14()
    elif action == 15:
        sousa.no15()
    elif action == 16:
        sousa.no16()
    elif action == 17:
        sousa.no17()
    elif action == 18:
        sousa.no18()
    elif action == 19:
        sousa.no19()
    elif action == 20:
        sousa.no20()
    elif action == 21:
        sousa.no21()
    elif action == 22:
        sousa.no22()

puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)
import time

DqnAgent = DQNAgent()
FIELD_LABELS = 6
NEXT_LABELS = 4

EPISODE = 50

def main():
    win_count = 0
    lose_count = 0
    field = np.zeros((12,6,6))
    next1 = np.zeros((2,4))
    next2 = np.zeros((2,4))
    DqnAgent.get_action([field.reshape(1,12,6,6), field.reshape(1,12,6,6), next1.reshape(1,2,4), next2.reshape(1,2,4)])
    capture = cv2.VideoCapture(1)

    if (capture.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました") 
    count = 0
    ret, img = capture.read()
    count_time = 0
    print("Ready")
    while True:
        win_flag = False
        lose_flag = False
        q1 = collections.deque([], 4)
        q2 = collections.deque([], 4)
        fields = collections.deque([], 2)
        dodai_fields = collections.deque([], 2)
        nexts = collections.deque([], 2)
        scores = collections.deque([], 2)
        next_puyos = get_next_puyo_info(img)
        one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
        nexts.append(one_hot_next)
        if start_judge(img):
            while True:
                count_time += 1
                if count_time % 30 == 0:
                    #get_score(img)
                    count_time = 0
                win_flag = win_judge(img)
                lose_flag = lose_judge(img)
                if win_flag or lose_flag:
                    puyo_cont.clear()
                    print("finish")
                    count += 1
                    break
                player1_next = img[85 : 110 , 240 : 260]
                player1_next_next = img[142 : 162 , 259 : 274]
                player1_next = cv2.cvtColor(player1_next, cv2.COLOR_BGR2GRAY)
                player1_next_next = cv2.cvtColor(player1_next_next, cv2.COLOR_BGR2GRAY)
                q1.append(player1_next)
                q2.append(player1_next_next)
                if len(q1) == 4:
                    flag1 = (np.array_equal(q1[0], q1[2]) == 0) and (np.array_equal(q2[0], q2[2]) == 0)
                    flag2 = (np.array_equal(q1[1], q1[3]) == 1) and (np.array_equal(q2[1], q2[3]) == 1)

                    if flag1 and flag2:
                        field_puyos = get_field_info(img)
                        dodai_fields.append(field_puyos[0][8:])
                        one_hot_field = np.array(np.eye(FIELD_LABELS)[field_puyos])
                        fields.append(one_hot_field)
                        next_puyos = get_next_puyo_info(img)
                        one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
                        nexts.append(one_hot_next)
                        action = DqnAgent.get_action([one_hot_field[0].reshape(1,12,6,6), one_hot_field[1].reshape(1,12,6,6), nexts[0][0].reshape(1,2,4), nexts[0][1].reshape(1,2,4)])
                        try_action(action+1)

                        print(action)
                        q1.clear()
                        q2.clear()
                    else:
                        if (np.array_equal(q1[1], q1[3]) == 1) and (np.array_equal(q2[1], q2[3]) == 1):
                            direct.press('s')
                ret, img = capture.read()
        else:
            ret, img = capture.read()
            continue

        if win_flag:
            print('win')
            win_count += 1
        else:
            print('lose')
            lose_count += 1
        if count == EPISODE:
            break


        ret, img = capture.read()
    time.sleep(10)
    direct.press('esc')

    print(str(win_count) + " " + str(lose_count))

    
if __name__ == "__main__":
    main()
    