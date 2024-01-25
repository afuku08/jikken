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
    x = Conv2D(filters=16,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(my_puyo_input)
    x = Flatten()(my_puyo_input)

    enemy_puyo_input = Input(shape=(12,6,6),name='enemy_net')
    #y = Conv2D(filters=1,kernel_size = (12,1),strides=(1,1),activation='relu',padding='valid')(enemy_puyo_input)
    y = Flatten()(enemy_puyo_input)

    nowpuyo_input = Input(shape=(2, 4),name='nowpuyo_input')
    nextpuyo_input = Input(shape=(2, 4), name='nextpuyo_input')
    a = Flatten()(nowpuyo_input)
    b = Flatten()(nextpuyo_input)

    x = keras.layers.concatenate([x,a,b], axis=1)
    x = Dense(1000,activation='relu')(x)
    x = Dense(500,activation='relu')(x)


    x = keras.layers.concatenate([x,y],axis=1)
    x = Dense(1000,activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    output = Dense(22,activation='linear',name='output')(x)
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[my_puyo_input,enemy_puyo_input,nowpuyo_input,nextpuyo_input],outputs=output)
    plot_model(model, to_file='model.png',show_shapes=True)

    return model

create_new_Qmodel()