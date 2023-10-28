# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:01:04 2023

@author: candy
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

data = []
labels = []

puyo = ["akapuyo", "aopuyo", "midoripuyo", "kipuyo", "murasakipuyo", "ojamapuyo", "kara"]

for i in range(len(puyo)):
    puyo_file_name_list = os.listdir("./puyos/"+puyo[i])
    for j in range(0,len(puyo_file_name_list)-1):
        n=os.path.join("./puyos/"+puyo[i]+"/", puyo_file_name_list[j])
        img = cv2.imread(n)
        img = cv2.resize(img, (64, 64))
        data.append(img)
        labels.append(i)

data = np.array(data)
labels = np.array(labels)
shuffle_indices = np.random.permutation(len(data))
data = data[shuffle_indices]
labels = labels[shuffle_indices]

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

y_train = to_categorical(labels)
y_test = to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("sigmoid"))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model = load_model('my_model.h5')

history = model.fit(data, y_train, batch_size=128, 
                    epochs=50, verbose=1, validation_data=(X_test, y_test))


print(history.history)

score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#acc, val_accのプロット
plt.plot(history.history["accuracy"], label="accuracy", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_accuracy", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()


model.save("my_model.h5")