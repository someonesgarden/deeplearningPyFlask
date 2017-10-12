#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask') # 深い別のフォルダから指定フォルダのlibを読み出す場合
from app.model.Perceptron import Perceptron
from matplotlib.colors import ListedColormap

def main():

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases/iris/iris.data', header=None)
    T = df.iloc[0:100, 4].values
    T = np.where(T == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0,2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, T)

    # Plot graphs (1 row x 2 columns)
    fig = plt.figure(1) # グラフを保存する用にfigにキャッシュさせる
    plt.subplot(2, 2, 1) # 一つ目(左）
    plt_scatter(X)  # データの散布状態

    plt.subplot(2, 2, 2)  # 二つ目(右）
    plt_error(range(1, len(ppn.errors_)+1), ppn.errors_)  # 学習の遷移

    plt.subplot(2,2, 3)
    plot_decision_regions(X, T, classifier=ppn)

    plt.show()     # グラフ表示
    fig.savefig('data/output/perceptron.png')     # グラフ保存


# エポックと誤分類誤差の関係の折れ線グラフ
def plt_error(A, B):
    plt.plot(A, B, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

# データの散布図
def plt_scatter(X):
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s','x','o','^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() -1,  X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() -1,  X[:, 1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)



if __name__ == '__main__':
    main()