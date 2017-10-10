#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import os
import sys
path2proj = "/Users/user/PycharmProjects/deeplearningFlask"
sys.path.append(path2proj)

import struct
import pickle
import pyprind
import numpy as np
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from app.neuralnet import NeuralNetMLP
from app.neuralnet import MLPGradientCheck

cur_dir = os.path.dirname(__file__)
dest = os.path.join(cur_dir, 'krasapp/pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)


def load_mnist(path, kind='train'):
    """ Load MNIST Data """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    """ Load file """
    with open(labels_path, 'rb') as lbpath:
        # Exchange binary into chars
        # ８バイトのバイナリデータを指定、マジックナンバとアイテムの個数を読み込む
        magic, n = struct.unpack('>II', lbpath.read(8))
        # ファイルからラベルを読み込み配列を構築：fromfile関数の引数にファイルと配列のデータ形式を指定
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


class Keras_MNIST_App(object):

    def __init__(self):
        theano.config.floatX = 'float32'
        self.X_train, self.y_train = load_mnist('data/mnist', kind='train')
        self.X_test, self.y_test = load_mnist('data/mnist', kind='t10k')
        self.model = None

        print(self.X_train.shape)
        print(self.X_test.shape)
        print('Train > Rows: %d, columns: %d' % (self.X_train.shape[0], self.X_train.shape[1]))
        print('Test > Rows: %d, columns: %d' % (self.X_test.shape[0], self.X_test.shape[1]))

    def main(self):

        # exchange to 32bit
        self.X_train = self.X_train.astype(theano.config.floatX)
        self.X_test = self.X_test.astype(theano.config.floatX)
        print('first three labels: ', self.y_train[:3])
        y_train_ohe = np_utils.to_categorical(self.y_train)
        print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])

        # keras, main!
        np.random.seed(1)

        classifier_pkl = os.path.join(dest, 'classifier.pkl')
        if not os.path.exists(classifier_pkl):

            self.model = Sequential()  # モデルの初期化

            # 一つ目の隠れ層
            self.model.add(
                Dense(input_dim=self.X_train.shape[1],  # 入力ユニット数
                      output_dim=50,                    # 出力ユニット数
                      init='uniform',                   # 重みを一様乱数で初期化
                      activation='tanh'                 # 活性化関数（双曲線正接関数）
                      )
            )
            # 二つ目の隠れ層
            self.model.add(
                Dense(input_dim=50,
                      output_dim=50,
                      init='uniform',
                      activation='tanh')
            )
            # 出力層
            self.model.add(
                Dense(input_dim=50,
                      output_dim=y_train_ohe.shape[1],
                      init='uniform',
                      activation='softmax')
            )

            # モデルコンパイル時のオプティマイザの設定
            # 引数に学習率、荷重減衰定数、モーメンタム学習を設定
            sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)

            # モデルのコンパイル
            self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

            print("mnist keras classifier_pkl NOT exist")
            self.model.fit(self.X_train, y_train_ohe, nb_epoch=50, batch_size=300, verbose=1, validation_split=0.1)
            pickle.dump(self.model, open(os.path.join(dest, 'classifier.pkl'), 'wb'))

        else:
            print("mnist keras classifier_pkl exist")
            self.model = pickle.load(open(classifier_pkl, 'rb'))

    def check_acc(self):
        y_train_pred = self.model.predict_classes(self.X_train, verbose=0)
        print('First 3 predictions:', y_train_pred[:3])

        train_acc = float(np.sum(self.y_train == y_train_pred, axis=0)) / self.X_train.shape[0]
        print('Training accuracy: %.2f%%' % (train_acc * 100))

        y_test_pred = self.model.predict_classes(self.X_test, verbose=0)
        test_acc = float(np.sum(self.y_test == y_test_pred, axis=0)) / self.X_test.shape[0]
        print('Test accuracy: %.2f%%' % (test_acc * 100))