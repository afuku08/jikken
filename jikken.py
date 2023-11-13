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
                #cv2.imshow('banmen', grid)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                puyo = classifier.predict(grid, template_type="field")
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
    
def get_next_puyo_info(img):
    player1_next = img[73 : 123 , 240 : 260]
    h, w, c = player1_next.shape
    player1_next = player1_next[: h // 2], player1_next[h // 2 :]
    player1_next = [classifier.predict(i, template_type="p1") for i in player1_next]

    player1_next_next = img[132 : 172 , 259 : 274]
    h, w, c = player1_next_next.shape
    player1_next_next = player1_next_next[: h // 2], player1_next_next[h // 2 :]
    player1_next_next = [
        classifier.predict(i, template_type="p1") for i in player1_next_next
    ]

    player2_next = img[73 : 123 , 377 : 397]
    h, w, c = player2_next.shape
    player2_next = player2_next[: h // 2], player2_next[h // 2 :]
    player2_next = [classifier.predict(i, template_type="p2") for i in player2_next]

    player2_next_next = img[132 : 172 , 364 : 379]
    h, w, c = player2_next_next.shape
    player2_next_next = player2_next_next[: h // 2], player2_next_next[h // 2 :]
    player2_next_next = [
        classifier.predict(i, template_type="p2") for i in player2_next_next
    ]
    player1_nexts = [player1_next, player1_next_next]
    #player2_nexts = [player2_next, player2_next_next]
    #output = [player1_nexts, player2_nexts]
    return player1_nexts

class puyo_classifier(object):
    def __init__(self, puyo_types):
        self._puyo_types = puyo_types
        self._field_template = {}
        for name in self._puyo_types:
            img = cv2.resize(cv2.imread(f"images/field/{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._field_template[name] = img

        self._p1_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p1/p1_{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p1_template[name] = img

        self._p2_template = {}
        for name in self._puyo_types[:-2]:
            img = cv2.resize(cv2.imread(f"images/p2/p2_{name}.jpg"), (40, 40))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._p2_template[name] = img

    def predict(self, img, template_type="field"):
        img = cv2.resize(img, (40, 40))
        differences = []

        channel = 0

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

        for name in self._puyo_types[:-2]:
            if template_type == "field":
                template_img = self._field_template[name]
            elif template_type == "p1":
                template_img = self._p1_template[name]
            elif template_type == "p2":
                template_img = self._p2_template[name]
            template_img_hist = cv2.calcHist(
                [template_img], [channel], None, [256], [0, 256]
            )

            diff = cv2.compareHist(template_img_hist, img_hist, 0)
            differences.append(diff)

        if template_type == "field":
            channel = 0
            img_hist = cv2.calcHist([img], [channel], None, [256], [0, 256])

            for name in ["ojama", "back"]:
                satu = np.mean(img[:, :, 1])
                if satu < 110 and name == "back":
                    diff = 1.00
                else:
                    template_img = self._field_template[name]
                    template_img_hist = cv2.calcHist(
                        [template_img], [channel], None, [256], [0, 256]
                    )
                    diff = cv2.compareHist(template_img_hist, img_hist, 0)

                differences.append(diff)
        puyo_type = differences.index(max(differences))
        return puyo_type
    
def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field
                
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
    #cv2.imshow('banmen', img[180:300, 223:403])
    now_hist = cv2.calcHist([img[180:300, 223:403]], [2], None, [256], [0, 256])
    comp_percent = cv2.compareHist(go_hist, now_hist, 0)
    #print(comp_percent)
    if comp_percent > 0.3:
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

    def add(self, experience):
        self.buffer.append(experience)

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

def create_Qmodel(learning_rate = 0.1**(4)):

    puyo_input = Input(shape=(12,6,7),name='puyo_net')
    x = Conv2D(filters=1,kernel_size = (12,1),strides=(1,1),activation='relu',padding='valid')(puyo_input)
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
    #plot_model(model, to_file='model.png',show_shapes=True)

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
        self.qnet = create_Qmodel(self.lr)
        self.qnet_target = create_Qmodel(self.lr)
    
    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet.predict(state)
            return np.argmax(qs[0])
        
    def learning(self,batch_size=32):
        if self.replay_buffer.len() <= self.batch_size:
            return

        inputs = np.zeros((batch_size,12,6,7))
        inputs_puyo0 = np.zeros([batch_size, 2, 5])
        inputs_puyo1 = np.zeros([batch_size, 2, 5])
        #inputs_puyo2 = np.zeros([batch_size, 2, 5])
        targets = np.zeros((batch_size,self.action_size))
        mini_batch = self.replay_buffer.sample(batch_size)

        for i,(state_b,puyos_b,action_b,reward_b,next_state_b,next_puyos_b) in enumerate(mini_batch):
            #state_b = stage2Binary(next_state_b)
            inputs[i:i+1] = state_b #　盤面
            inputs_puyo0[i:i+1] = puyos_b[0]
            inputs_puyo1[i:i+1] = puyos_b[1]
            #inputs_puyo2[i:i+1] = puyos_b[2]

            target = reward_b #　state_b盤面の時action_bを行って得た報酬
            cd = next_state_b == np.zeros(state_b.shape).all(axis=1)

            if not cd.all(): # 次状態の盤面が全て0でないなら
                #next_state_b = stage2Binary(next_state_b)
                neMap = map2batch(next_state_b)
                retMainQs = self.qnet.predict([neMap,next_puyos_b[0].reshape(1,2,5),next_puyos_b[1].reshape(1,2,5)])[0]
                next_action = np.argmax(retMainQs)
                target = reward_b + self.gamma * self.qnet_target.predict([next_state_b,next_puyos_b[0].reshape(1,2,5),next_puyos_b[1].reshape(1,2,5)])[0][next_action]
                if target < -1:
                    target = -1

            targets[i] = self.qnet.predict([state_b,puyos_b[0].reshape(1,2,5),puyos_b[1].reshape(1,2,5)])
            targets[i][action_b] = target
        self.qnet.fit([inputs,inputs_puyo0,inputs_puyo1], targets, epochs=1, verbose=0)

        return self.qnet

def map2batch(gameMap,batch_size = 1):
    return gameMap.reshape((batch_size,13,6,5))

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
        sousa.no513()
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
FIELD_LABELS = 7
NEXT_LABELS = 5

def main():
    field = np.zeros((12,6,7))
    next1 = np.zeros((2,5))
    next2 = np.zeros((2,5))
    #ans = qnet.predict([field.reshape(1,12,6,7), next1.reshape(1,2,5), next2.reshape(1,2,5)])
    DqnAgent.get_action([field.reshape(1,12,6,7), next1.reshape(1,2,5), next2.reshape(1,2,5)])
    capture = cv2.VideoCapture(1)
    #capture.set(cv2.CAP_PROP_FPS, 60)

    if (capture.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました") 
    count = 0
    ret, img = capture.read()
    #for i in range(50):
    count_time = 1
    while True:
        #start = time.time()
        win_flag = False
        lose_flag = False
        #arr1 = []
        #arr2 = []
        q1 = collections.deque([], 5)
        q2 = collections.deque([], 5)
        fields = collections.deque([], 2)
        nexts = collections.deque([], 2)
        scores = collections.deque([], 2)
        #next2s = collections.deque([], 2)
        next_puyos = get_next_puyo_info(img)
        nexts.append(next_puyos[0])
        #next2s.append(next_puyos[1])
        results[0] = 0
        results[1] = 0
        if start_judge(img):
            while True:
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

                    if flag1 and flag2:
                        scores.append(results)
                        #print('tumo')
                        field_puyos = get_field_info(img)
                        one_hot_field = np.array(np.eye(FIELD_LABELS)[field_puyos])
                        #one_hot_field.append(np.eye(FIELD_LABELS)[field_puyos])
                        fields.append(one_hot_field)
                        next_puyos = get_next_puyo_info(img)
                        one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
                        #one_hot_next.append(np.eye(NEXT_LABELS)[next_puyos[0]])
                        nexts.append(one_hot_next)
                        action = DqnAgent.get_action([one_hot_field[0].reshape(1,12,6,7), one_hot_next[0].reshape(1,2,5), one_hot_next[1].reshape(1,2,5)])
                        try_action(action+1)
                        if len(fields) == 2:
                            reward = scores[0][0] - scores[0][1];
                            DqnAgent.replay_buffer.add((fields[0], nexts[0], action, reward, fields[1], nexts[1]))

                        print(action)
                        q1.clear()
                        q2.clear()
                #print(1)
                ret, img = capture.read()
                end = time.time()
                print(end - start)
        else:
            #print(time.time() - start)
            ret, img = capture.read()
            continue
        if count == 5:
            break
        DqnAgent.learning()
        ret, img = capture.read()

    
if __name__ == "__main__":
    main()
    