#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
PERCEPTRON
USING Linear Regression
'''

import numpy as np
import pandas as pd


class LinearRegression(object):
    def __init__(self, learning_rate=0.01, EPOCH=100):
        self.learning_rate = learning_rate
        self.EPOCH = EPOCH

    def fit(self, X, T):
        N,D = X.shape
        self.errors_ = []

        #self.w = np.zeros(D+1)
        self.w = np.random.randn(D + 1)

        for _ in range(self.EPOCH):
            errors = 0
            for xi, target in zip(X, T):
                update = self.learning_rate * (target - self.predict(xi))
                self.w[0] += update * 1
                self.w[1:] += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return X.dot(self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

