#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import importlib
import os.path
importlib.reload(sys)

cur_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(cur_dir, '../../static/uploaded/boston_house_prices.csv'), header=0)
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']


# def plot_pair():
#     sns.set(style='whitegrid', context='notebook')
#     sns.pairplot(df[cols], size=2.0)
#     plt.show()
#
#
# def plot_corrcoef():
#     cm = np.corrcoef(df[cols].values.T)
#     sns.set(font_scale=1.5)
#     hm =sns.heatmap(cm,
#                     cbar=True,
#                     annot=True,
#                     square=True,
#                     fmt='.2f',
#                     annot_kws={'size':15},
#                     yticklabels=cols,
#                     xticklabels=cols)
#     plt.show()


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0:] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

