#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask') # 深い別のフォルダから指定フォルダのlibを読み出す場合

from app.model.ADAlineSGD import *
from app.lib.util import plot_decisionregions
from app.lib.util import Xstdardize

from sklearn import datasets


def main():
    argvs = sys.argv
    argc = len(argvs)

    mode = argvs[1]
    print(mode == 'sk')


    if mode == 'sk':
        iris = datasets.load_iris()
        X = iris.data[:, [2,3]]
        T = iris.target

    else:
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        y = df.iloc[0:100,4].values
        T = np.where(y == 'Iris-setosa', 1, -1)
        X = df.iloc[0:100, [0,2]].values
        X = Xstdardize(X)

    ### FIT
    ada = ADAlineSGD(eta=0.01, n_iter=20, shuffle=1, random_state=1)
    ada.partial_fit(X , T)
    ada2 = ADAlineSGD(eta=0.01, n_iter=20, shuffle=1, random_state=1)
    ada2.fit(X, T)

    ### PLOT
    resolution = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = ada2.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    ### SUBPLOT

    fig = plt.figure("ADAline SGD", figsize=(12, 5), facecolor='white')

    plt.subplot(1, 2, 1)

    plt.scatter(X[:50,0], X[:50,1],color='red', marker='o')
    plt.scatter(X[50:,0], X[50:,1],color='blue',marker='x')

    #plt_decision_regions(X, y, ada)
    plot_decisionregions(X, X1, X2, Z, T)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.title('Scatter Plot / decisionregions')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    plt.xlabel('n_iters')
    plt.ylabel('log(Sum-squared-error)')
    plt.title('Adaline - Learning rate {0}'.format(0.1))
    plt.show()

    fig.savefig("data/output/adalineSGD.png")

if __name__ == '__main__':
    main()