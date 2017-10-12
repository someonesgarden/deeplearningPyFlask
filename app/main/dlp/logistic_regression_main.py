#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.model.LogisticRegression import *
import sklearn.datasets as ds
from app.lib.util import plot_decisionregions

def main():

    # df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    #                  'machine-learning-databases/iris/iris.data', header=None)
    # y = df.iloc[0:100, 4].values
    # # Sigmoidの場合は [0,1]
    # # Linear Regressionやtanhの場合は、[-1,1]
    # T = np.where(y == 'Iris-setosa', 1, 0)
    # X = df.iloc[0:100, [0, 2]].values

    iris = ds.load_iris()
    X = iris.data[:100, [0, 2]]
    T = iris.target[:100]

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o')
    plt.scatter(X[50:, 0], X[50:, 1], color='blue', marker='x')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.show()

    ###
    lrn = LogisticRegression(learning_rate=0.01, EPOCH=20)
    lrn.fit(X, T)
    plt.plot(range(1, len(lrn.errors_) + 1), lrn.errors_)
    plt.show()

    ###
    resolution = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    Xb = np.array([X1.ravel(), X2.ravel()]).T
    I = np.array([[1] * Xb.shape[0]]).T
    Xb = np.concatenate((I, Xb), axis=1)

    #print Xb.shape
    Z = lrn.predict(np.array(Xb))
    Z = Z.reshape(X1.shape)

    plot_decisionregions(Xb, X1, X2, Z, T)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

if __name__=='__main__':
    main()