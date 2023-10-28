# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:33:09 2023

@author: candy
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

puyo = ["akapuyo", "aopuyo", "midoripuyo", "kipuyo", "murasakipuyo", "ojamapuyo", "kara"]

X_test = []
Y_test = []

for i in range(len(puyo)):
    puyo_file_name_list = os.listdir("./puyos/"+puyo[i])
    for j in range(0,len(puyo_file_name_list)-1):
        n=os.path.join("./puyos/"+puyo[i]+"/", puyo_file_name_list[j])
        img = cv2.imread(n)
        img = cv2.resize(img, (64, 64))
        X_test.append(img)
        Y_test.append(i)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
shuffle_indices = np.random.permutation(len(X_test))
X_test = X_test[shuffle_indices]
Y_test = Y_test[shuffle_indices]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense 
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model
y_test = to_categorical(Y_test)

model = load_model('my_model.h5')

test = cv2.imread("puyos/trakara/puyo_tra_1_3.jpg")
test = cv2.resize(test, (64, 64))
test = np.expand_dims(test, axis=0)
print(model.predict(test))
nameNumLabel=np.argmax(model.predict(test))
print(nameNumLabel)

'''score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()'''
