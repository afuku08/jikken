# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:04:32 2023

@author: candy
"""

class CNN:
    """ 予測モデルを作成する関数
    引数:
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder
      image_size: 画像サイズ
      num_clesses:識別数
    返り値:
      y_conv: 各クラスの確率の配列 ([tensorflow.python.framework.ops.Tensor]型)
    """
    def makeMnistCNN(imegs_placeholder, keep_prob , image_size , num_clesses):
        # 重みを初期化
        def weight_variable(shape):
            # 重みを標準偏差0.1の正規分布で初期化
            inital = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(inital)

        # バイアスを初期化
        def bias_variable(shape):
            # 定数0.0で初期化
            inital = tf.constant(0.0, shape=shape)
            return tf.Variable(inital)

        # 畳み込み層
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

        # プーリング層
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


        # 入力層を28*28*1に変形
        x_image = tf.reshape(imegs_placeholder, [-1, image_size, image_size, 1])

        # 畳み込み層1の作成
        with tf.name_scope("conv1") as scope:
            W_conv1 = weight_variable([3,3,1,16])
            b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # プーリング層1の作成
        with tf.name_scope("pool1") as scope:
            h_pool1 = max_pool_2x2(h_conv1)

        # 畳み込み層2の作成
        with tf.name_scope("conv2") as scope:
            W_conv2 = weight_variable([3,3,16,32])
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # プーリング層2の作成
        with tf.name_scope("pool2") as scope:
            h_pool2 = max_pool_2x2(h_conv2)

        # 畳み込み層3の作成
        with tf.name_scope("conv3") as scope:
            W_conv3 = weight_variable([3,3,32,32])
            b_conv3 = bias_variable([32])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

        # 畳み込み層4の作成
        with tf.name_scope("conv4") as scope:
            W_conv4 = weight_variable([3,3,32,32])
            b_conv4 = bias_variable([32])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

        # 結合層1の作成
        with tf.name_scope("fc1") as scope:
            # 1次元に変形
            h_flat = tf.reshape(h_conv4, [-1, 7*7*32])
            W_fc1 = weight_variable([7*7*32, 512])   
            b_fc1 = bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

            # dropout1の設定
            h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 結合層2の作成
        with tf.name_scope("fc2") as scope:
            W_fc2 = weight_variable([512, num_clesses])
            b_fc2 = bias_variable([num_clesses])

        # ソフトマックス関数による正規化
        with tf.name_scope("softmax") as scope:
            y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

        return y_conv
