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
    #cv2.imshow('banmen', img[180:300, 223:403])
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

puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)
import time

DqnAgent = DQNAgent()
FIELD_LABELS = 7
NEXT_LABELS = 5

def main():
    capture = cv2.VideoCapture(1)

    if (capture.isOpened()== False):  
        print("ビデオファイルを開くとエラーが発生しました") 
    count = 0
    ret, img = capture.read()
    #for i in range(50):
    while True:
        win_flag = False
        lose_flag = False
        #arr1 = []
        #arr2 = []
        q1 = collections.deque([], 3)
        q2 = collections.deque([], 3)
        fields = collections.deque([], 2)
        nexts = collections.deque([], 2)
        #next2s = collections.deque([], 2)
        next_puyos = get_next_puyo_info(img)
        nexts.append(next_puyos[0])
        #next2s.append(next_puyos[1])
        if start_judge(img):
            while True:
                start = time.time()
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
                if len(q1) == 3:
                    flag1 = (np.array_equal(q1[0], q1[1]) == 0) and (np.array_equal(q2[0], q2[1]) == 0)
                    flag2 = (np.array_equal(q1[1], q1[2]) == 1) and (np.array_equal(q2[1], q2[2]) == 1)

                    if flag1 and flag2:
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
                        print(action)
                        q1.clear()
                        q2.clear()
                print(1)
                ret, img = capture.read()
                end = time.time()
                print(end - start)
        else:
            ret, img = capture.read()
            continue
        if count > 50:
            break
        print(0)
        ret, img = capture.read()

    
if __name__ == "__main__":
    main()
    