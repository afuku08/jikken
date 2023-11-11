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
    h_start = 71
    player1_field = img[h_start : h_start + H, 90 : 90 + W]
    player2_field = img[h_start : h_start + H, 414 : 414 + W]
    fields = []
    for field in [player1_field, player2_field]:
        init_field = np.zeros((12, 6), dtype=np.uint8)

        h_unit = H // 12
        w_unit = W // 6
        for h in range(0, H, h_unit):
            for w in range(0, W, w_unit):
                grid = field[h : h + h_unit, w : w + w_unit]
                puyo = classifier.predict(grid, template_type="field")
                init_field[h // h_unit, w // w_unit] = puyo

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
    
def puyo_class(img):
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    return np.argmax(model.predict(img))

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

def field_edit(field):
    if field[1, 2] == 6:
        field[0, 2] = 6
    return field

import time

puyo_types = ["aka", "ao", "kiiro", "midori", "murasaki", "ojama", "back"]
classifier = puyo_classifier(puyo_types)

FIELD_LABELS = 7
NEXT_LABELS = 5

qnet = create_Qmodel()
field = np.zeros((12,6,7))
next1 = np.zeros((2,5))
next2 = np.zeros((2,5))
fields = collections.deque([], 2)
nexts = collections.deque([], 2)
start = time.time()
ans = qnet.predict([field.reshape(1,12,6,7), next1.reshape(1,2,5), next2.reshape(1,2,5)])
print(time.time() - start)
print(np.argmax(ans[0]))
start = time.time()
ans = qnet.predict([field.reshape(1,12,6,7), next1.reshape(1,2,5), next2.reshape(1,2,5)])
print(time.time() - start)
print(np.argmax(ans[0]))
img = cv2.imread("banmen1.png")
field_puyos = get_field_info(img)
one_hot_field = np.array(np.eye(FIELD_LABELS)[field_puyos])
#one_hot_field.append(np.eye(FIELD_LABELS)[field_puyos])
fields.append(one_hot_field)
next_puyos = get_next_puyo_info(img)
one_hot_next = np.array(np.eye(NEXT_LABELS)[next_puyos])
#one_hot_next.append(np.eye(NEXT_LABELS)[next_puyos[0]])
nexts.append(one_hot_next)
action = qnet.predict([one_hot_field[0].reshape(1,12,6,7), one_hot_next[0].reshape(1,2,5), one_hot_next[1].reshape(1,2,5)])
print(action)
