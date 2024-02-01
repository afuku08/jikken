import keras
from tensorflow.keras.layers import Input
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import concatenate,add
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape,Permute# モジュールのインポート
from tensorflow.keras.layers import Conv2D,Convolution2D, MaxPooling2D,Cropping2D,Conv2DTranspose# CNN層、Pooling層のインポート
from keras.optimizers import Adam
from keras.utils import plot_model

def create_new_Qmodel(learning_rate = 0.1**(4)):
    my_puyo_input = Input(shape=(12,6,6),name='puyo_net')
    x = Conv2D(filters=256,kernel_size = (2,2),padding='same',activation='relu',)(my_puyo_input)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=256,kernel_size = (2,2),padding='same',activation='relu',)(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    enemy_puyo_input = Input(shape=(12,6,6),name='enemy_net')
    y = Conv2D(filters=256,kernel_size = (2,2),padding='same',activation='relu',)(enemy_puyo_input)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=256,kernel_size = (2,2),padding='same',activation='relu',)(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)

    nowpuyo_input = Input(shape=(2, 4),name='nowpuyo_input')
    nextpuyo_input = Input(shape=(2, 4), name='nextpuyo_input')
    a = Flatten()(nowpuyo_input)
    b = Flatten()(nextpuyo_input)

    x = keras.layers.concatenate([x,y,a,b], axis=1)

    x = Dense(400, activation='relu')(x)
    output = Dense(22,activation='softmax',name='output')(x)
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[my_puyo_input,enemy_puyo_input,nowpuyo_input,nextpuyo_input],outputs=output)
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    plot_model(model, to_file='model.png',show_shapes=True)

    return model

create_new_Qmodel()