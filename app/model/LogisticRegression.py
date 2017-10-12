#!/usr/bin/env python3
# -*- coding: UTF-8 -*-# enable debugging

'''
PERCEPTRON
Logistic Regression using "SIGMOID"
'''

import numpy as np
from app.lib.util import sigmoid, sigmoid_cost, cross_entropy
from app.lib.plots import showScatter

class LogisticRegression(object):
    def __init__(self, learning_rate=0.01, EPOCH=100):
        self.learning_rate = learning_rate
        self.EPOCH = EPOCH

    def fit(self, X, T):
        N, D = X.shape
        self.errors_ = []

        I = np.array([[1] * N]).T
        Xb = np.concatenate((I, X), axis=1)
        #self.w = np.zeros(D + 1)
        self.w = np.random.randn(D + 1)
        z = Xb.dot(self.w)
        Y = sigmoid(z)

        for i in range(self.EPOCH):
            # if i % 10 ==0:
            print(sigmoid_cost(T, Y))

            # gradient descent weight update
            update = self.learning_rate * np.dot((T - Y).T, Xb)
            self.w += update

            errors = np.abs(update).sum()
            self.errors_.append(errors)

            # recalculate Y
            Y = sigmoid(Xb.dot(self.w))

        print("Final w:", self.w)

    def predict(self,X):
        return sigmoid(X.dot(self.w))