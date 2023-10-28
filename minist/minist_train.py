# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:05:26 2023

@author: candy
"""

print('mnist_train.py START')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime

import DeepConvNet as CNN

IMAGE_SIZE  = 28    # 画像サイズ
NUM_CLASSES = 10    # 識別数

print('MNIST Download Start')
# MNISTデータのダウンロード
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print('MNIST Download End')

""" 損失関数
    引数:
      logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
      labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
      cross_entropy: 交差エントロピーのtensor, float
"""
def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    return cross_entropy

""" 訓練のopを定義する関数
    引数:
      loss: 損失のtensor, loss()の結果
      learning_rate: 学習係数
    返り値:
      train_step: 訓練のop
"""
def training(loss, learning_rate):
    # 勾配降下法(Adam)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

"""正解率(accuracy)を計算する関数
    引数:
        logits: ロジットのtensor, float - [batch_size, NUM_CLASSES]
        labels: ラベルのtensor, int32 - [batch_size, NUM_CLASSES]
    返り値:
        accuracy: 正解率(float)
"""
def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy

if __name__=="__main__":
    with tf.Graph().as_default():
        print('設定 START') 
        x_image = tf.placeholder("float", shape=[None, IMAGE_SIZE * IMAGE_SIZE])    # 入力
        y_label = tf.placeholder("float", shape=[None, NUM_CLASSES]) # 出力
        keep_prob = tf.placeholder("float")    #ドロップアウト

        # モデルを作成
        logits = CNN.CNN.makeMnistCNN(x_image, keep_prob , IMAGE_SIZE , NUM_CLASSES) 

        # opを定義
        loss_value = loss(logits, y_label) 
        train_op = training(loss_value,1e-4) 
        accur = accuracy(logits, y_label) 

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #TensorBoardで追跡する変数を定義
        accuracy_op_train = tf.summary.scalar("Accuracy on Train", accur)
        accuracy_op_test = tf.summary.scalar("Accuracy on Test", accur)
        summary_op_train = tf.summary.merge([accuracy_op_train])
        summary_op_test = tf.summary.merge([accuracy_op_test])
        summary_writer = tf.summary.FileWriter("./TensorBoard", graph=sess.graph)

        # 訓練したモデルを保存
        # (tf.train.Saver()が呼ばれる前までに呼ばれた引数が対象になる)
        saver = tf.train.Saver()
        print('設定 END')

        print('学習 START : ' + str(datetime.datetime.now()))
        #学習の実行
        for epoch in range(5000):
            #訓練データセットから 50 のランダムなデータの “バッチ” を取得 [0]に画像の配列、[1]に結果の配列
            batch = mnist.train.next_batch(50)

            # 学習の途中経過の表示・TensorBoard書き込み
            if epoch % 100 == 0:
                train_accury = sess.run(accur, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob: 1.0})

                # テストデータ(検証データ)で評価
                test_batch = mnist.validation.next_batch(500, shuffle=False)
                test_accury = sess.run(accur, feed_dict={x_image: test_batch[0], y_label: test_batch[1], keep_prob: 1.0})
                # ↓ Jupiterで実行するとコンソールが落ちる (メモリ不足？)
                #test_accury = sess.run(accur, feed_dict={x_image: mnist.validation.images, y_label: mnist.validation.labels, keep_prob: 1.0})
                print("epoch:%d, train_accury : %g  test_accury : %g"%(epoch, train_accury , test_accury))

                summary_str_train = sess.run(summary_op_train, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob: 1.0})
                summary_writer.add_summary(summary_str_train, epoch)

                summary_str_test = sess.run(summary_op_test, feed_dict={x_image: test_batch[0], y_label: test_batch[1], keep_prob: 1.0})
                #summary_str = sess.run(summary_op_test, feed_dict={x_image: mnist.validation.images, y_label: mnist.validation.labels, keep_prob: 1.0})
                summary_writer.add_summary(summary_str_test, epoch)
                summary_writer.flush()

            # 学習
            sess.run(train_op, feed_dict={x_image: batch[0], y_label: batch[1], keep_prob:0.5})

        print('学習 END : ' + str(datetime.datetime.now()))

        #結果表示 (テストデータで評価)
        test_batch = mnist.test.next_batch(500, shuffle=False)
        print("test accuracy : %g" %sess.run(accur, feed_dict={x_image: test_batch[0], y_label: test_batch[1], keep_prob: 1.0}))
        #print("test accuracy : %g" %sess.run(accur, feed_dict={x_image: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0}))

        save_path = saver.save(sess, "./ckpt/model.ckpt") # 変数データ保存
        print('Save END : ' + save_path )

        summary_writer.close()
        sess.close()

        print('mnist_train.py END')
