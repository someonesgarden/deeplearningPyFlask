#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")

from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from app.lib.util import decision_region_graph

def main(output=0):
    # DATA #30% test / 70% training data =====
    iris = ds.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # STANDARD ===============================
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # OUTPUT ==================================
    if output == 1:
        C_graph(X_train_std, y_train)
    else:
        #svm = SGDClassifier(loss='log')
        #svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm = SVC(kernel='rbf', C=10.0, random_state=0,gamma=0.45)
        decision_region_graph(X_train_std, y_train, X_test_std, y_test,svm)

def C_graph(X_train_std, y_train):

    # C = 1 / lambda =========================
    weights, params =[],[]
    for c in np.arange(-5,5,0.1):
        svm = SVC(kernel='linear', C=1.0**c, random_state=0)
        svm.fit(X_train_std, y_train)
        weights.append(svm.coef_[1])
        params.append(10**c)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label = 'petal length')
    plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()

if __name__ == "__main__":
    main()
