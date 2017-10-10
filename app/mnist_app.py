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
import matplotlib.pyplot as plt
from app.neuralnet import NeuralNetMLP
from app.neuralnet import MLPGradientCheck

cur_dir = os.path.dirname(__file__)
dest = os.path.join(cur_dir, 'mnist/pkl_objects')
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


def load_mnist_csv(path,kind='train'):
    labels_path = os.path.join(path, '%s_img.csv' % kind)
    labels = np.genfromtxt(labels_path, dtype=int, delimiter=',')
    images_path = os.path.join(path, '%s_labels.csv' % kind)
    images = np.genfromtxt(images_path, dtype=int, delimiter=',').reshape(len(labels), 784)
    return images, labels

# def save_data_csv():
#     np.savetxt(os.path.join('data/mnist', 'train_img.csv'), X_train, fmt='%i', delimiter=',')
#     np.savetxt(os.path.join('data/mnist', 'train_labels.csv'), y_train, fmt='%i', delimiter=',')
#     np.savetxt(os.path.join('data/mnist', 'test_img.csv'), X_test, fmt='%i', delimiter=',')
#     np.savetxt(os.path.join('data/mnist', 'test_labels.csv'), y_test, fmt='%i', delimiter=',')


class MNIST_App(object):

    def __init__(self):
        dest_train = os.path.join('data/mnist', 'train_img.csv')
        dest_test = os.path.join('data/mnist', 'test_img.csv')

        # if not os.path.exists(dest_train):
        #     self.X_train, self.y_train = load_mnist('data/mnist', kind='train')
        #     self.X_test, self.y_test = load_mnist('data/mnist', kind='t10k')
        #     save_data_csv()
        # else:
        #     X_train, y_train = load_mnist_csv('data/mnist', kind='train')
        #     X_test, y_test = load_mnist_csv('data/mnist', kind='test')

        self.X_train, self.y_train = load_mnist('data/mnist', kind='train')
        self.X_test, self.y_test = load_mnist('data/mnist', kind='t10k')

        self.nn = None

        print(self.X_train.shape)
        print(self.X_test.shape)
        print('Train > Rows: %d, columns: %d' % (self.X_train.shape[0], self.X_train.shape[1]))
        print('Test > Rows: %d, columns: %d' % (self.X_test.shape[0], self.X_test.shape[1]))

    def subplt_txtimgs(self, col=5, row=6):
        fig, ax = plt.subplots(nrows=row, ncols=col, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(int(col*row)):
            img = self.X_train[self.y_train == 7][i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    def main(self):

        print(self.X_train.shape[1])
        self.nn = NeuralNetMLP(n_output=10,
                          n_features=self.X_train.shape[1],
                          n_hidden=80,
                          l2=0.2,
                          l1=0.0,
                          epochs=1000,
                          eta=0.0008,
                          alpha=0.001,
                          decrease_const=0.00001,
                          shuffle=True,
                          minibatches=50,
                          random_state=1)

        # self.nn = MLPGradientCheck(n_output=10,
        #                        n_features=self.X_train.shape[1],
        #                        n_hidden=50,
        #                        l2=0.1,
        #                        l1=0.0,
        #                        epochs=1000,
        #                        eta=0.001,
        #                        alpha=0.001,
        #                        decrease_const=0.00001,
        #                        shuffle=True,
        #                        minibatches=50,
        #                        random_state=1)


        classifier_pkl = os.path.join(dest, 'classifier.pkl')

        if not os.path.exists(classifier_pkl):
            print("classifier_pkl NOT exist")
            self.nn.fit(self.X_train, self.y_train, print_progress=True)
            pickle.dump(self.nn,
                        open(os.path.join(dest, 'classifier.pkl'), 'wb')
                        )
        else:
            print("classifier_pkl exist")
            self.nn = pickle.load(open(classifier_pkl, 'rb'))

        y_train_pred = self.nn.predict(self.X_train)

        acc = float(np.sum(self.y_train == y_train_pred, axis=0)) / self.X_train.shape[0]
        print('Training accuracy: %.2f%%' % (acc * 100))

        batches = np.array_split(range(len(self.nn.cost_)), 1000)

        cost_ary = np.array(self.nn.cost_)
        cost_avgs = [np.mean(cost_ary[i]) for i in batches]

        #plt.plot(range(len(nn.cost_)), nn.cost_)
        plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
        plt.ylim([0, 2000])
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.show()

    def test_pred(self):

        y_test_pred = self.nn.predict(self.X_test)
        acc = float(np.sum(self.y_test == y_test_pred, axis=0)) / self.X_test.shape[0]
        print('Test accuracy: %.2f%%' % (acc * 100))

    # 正解した予想のケースを１００例
    def mnist_fit_correct(self):

        y_test_pred = self.nn.predict(self.X_test)
        misc1_img = self.X_test[self.y_test == y_test_pred][:100]
        correct_lab = self.y_test[self.y_test == y_test_pred][:100]
        misc1_lab = y_test_pred[self.y_test == y_test_pred][:100]
        fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(100):
            img = misc1_img[i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], misc1_lab[i]))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.show()

    # 外れた場合の予想のケースを100例
    def mnist_fit_wrong(self):

        y_test_pred = self.nn.predict(self.X_test)
        misc1_img = self.X_test[self.y_test != y_test_pred][:100]
        correct_lab = self.y_test[self.y_test != y_test_pred][:100]
        misc1_lab = y_test_pred[self.y_test != y_test_pred][:100]
        fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(100):
            img = misc1_img[i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], misc1_lab[i]))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.show()

    # とにかく100例
    def mnist_fit_all(self):

        y_test_pred = self.nn.predict(self.X_test)
        misc1_img = self.X_test[:][200:400]
        correct_lab = self.y_test[:][200:400]
        misc1_lab = y_test_pred[:][200:400]
        fig, ax = plt.subplots(nrows=20, ncols=10, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(200):
            img = misc1_img[i].reshape(28, 28)
            ax[i].imshow(img, cmap='Greys', interpolation='nearest')
            ax[i].set_title('%d) t: %d p: %d' % (i + 1, correct_lab[i], misc1_lab[i]))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.show()

