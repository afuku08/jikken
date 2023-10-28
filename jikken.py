import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import Input
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model
from collections import deque
import numpy as np
import cv2
import random

def get_field_info(img):
    H = 324
    W = 132
    h_start = 70
    player1_field = img[h_start : h_start + H, 91 : 91 + W]
    player2_field = img[h_start : h_start + H, 415 : 415 + W]
    fields = []
    for field in [player1_field, player2_field]:
        init_field = np.zeros((12, 6), dtype=np.uint8)

        h_unit = H // 12
        w_unit = W // 6
        
        puyo_cont = []
        
        for h in range(0, H, h_unit):
            for w in range(0, W, w_unit):
                grid = field[h : h + h_unit, w : w + w_unit]
                '''cv2.imshow('banmen', grid)
                cv2.waitKey(0)
                cv2.destroyAllWindows()'''
                puyo = puyo_class(grid)
                #print(puyo)
                this_puyo = -1
                #からの場合
                if puyo == 6:
                    this_puyo = 0
                #お邪魔ぷよの場合
                if puyo == 5:
                    this_puyo = 5
                #通常ぷよの場合
                else:
                    if len(puyo_cont) == 4:
                        this_puyo = 0
                    result = puyo not in puyo_cont
                    if result:
                        puyo_cont.append(puyo)
                        
                    this_puyo = puyo_cont.index(puyo) + 1
                    
                init_field[h // h_unit, w // w_unit] = this_puyo
                
        init_field = field_edit(init_field)
        fields.append(init_field)
    return fields
    
def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field
                
model = load_model('my_model.h5')

def puyo_class(img):
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    return np.argmax(model.predict(img))
    
class FieldConstructor(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (40, 40))
            self._field_template[name] = img

    def make_field_construct(self, field):
        init_img = np.zeros((480, 240, 3), dtype=np.uint8)
        for h in range(12):
            for w in range(6):
                puyo = field[h, w]
                puyo = self._puyo_types[int(puyo)]
                template_img = self._field_template[puyo]
                grid_h = h * 40
                grid_w = w * 40
                init_img[grid_h : grid_h + 40, grid_w : grid_w + 40] = template_img
        return init_img

go = cv2.imread('go.png')
go_hist = cv2.calcHist([go], [2], None, [256], [0, 256])
def start_judge(img):
    cv2.imshow('banmen', img[180:300, 223:403])
    now_hist = cv2.calcHist([img[180:300, 223:403]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(go_hist, now_hist, 0)
    if comp_percent > 0.4:
        return True
    else:
        return False

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

class Memory:
    def __init__(self,buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self,state, action, reward, next_state):
        data = (state, action, reward, next_state)
        self.buffer.append(data)

    def sample(self,batch_size):
        index = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
        return [self.buffer[i] for i in index]

    def len(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        return state, action, reward, next_state

class QNet(tf.keras.Model):
    def create_Qmodel(learning_rate = 0.1**(4)):

        puyo_input = Input(shape=(13,6,5),name='puyo_net')
        x = Conv2D(filters=1,kernel_size = (13,1),strides=(1,1),activation='relu',padding='valid')(puyo_input)
        x = Flatten()(x)

        y = Conv2D(filters=1,kernel_size = (1,6),strides=(1,1),activation='relu',padding='valid')(puyo_input)
        y = Flatten()(y)
        nowpuyo_input = Input(shape=(2, 5),name='nowpuyo_input')
        nextpuyo_input = Input(shape=(2, 5), name='nextpuyo_input')

        z = Conv2D(filters=16,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(puyo_input)
        z = Conv2D(filters=16,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
        z = MaxPooling2D()(z)
        z = Conv2D(filters=32,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
        z = Conv2D(filters=32,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
        z = MaxPooling2D()(z)   
        z = Flatten()(z)

        a = Flatten()(nowpuyo_input)
        b = Flatten()(nextpuyo_input)

        x = keras.layers.concatenate([x,y,z,a,b],axis=1)
        x = Dense(1000,activation='relu')(x)
        x = Dense(400, activation='relu')(x)
        output = Dense(22,activation='linear',name='output')(x)
        optimizer = Adam(lr=learning_rate)
        model = Model(inputs=[puyo_input,nowpuyo_input,nextpuyo_input],outputs=output)
        model.compile(optimizer=optimizer,loss='mean_squared_error')
        plot_model(model, to_file='model.png',show_shapes=True)

        return model

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 22

        self.replay_buffer = Memory(self.buffer_size, self.batch_size)
        self.qnet = QNet.create_Qmodel(self.lr)
        self.qnet_target = QNet.create_Qmodel(self.lr)
    
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.predict(state)
            return qs.data.argmax()

puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]

import time

def main():
    capture = cv2.VideoCapture(1)

    if (capture.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました") 

        ret, img = capture.read()

    for i in range(50):
        win_flag = False
        lose_flag = False
        if start_judge(img):
            while True:
                win_flag = win_judge(img)
                lose_flag = lose_judge(img)
                if win_flag or lose_flag:
                    break
                print(1)
                ret, img = capture.read()
    
        print(0)
        ret, img = capture.read()

    
if __name__ == "__main__":
    main()
    