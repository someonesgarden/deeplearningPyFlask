#!/usr/bin/env python3

'''
Random Forest
'''


# _*_ coding:utf-8 _*_

import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')

from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from app.lib.util import decision_region_graph
from sklearn.tree import export_graphviz
import os

def main(max_depth=3):

    # IRIS
    # iris = ds.load_iris()
    # X = iris.data[:, [2, 3]]
    # y = iris.target
    #

    # XOR
    # np.random.seed(0)
    # X_xor = np.random.randn(200, 2)
    # y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    # y_xor = np.where(y_xor, 1, -1)
    # X = X_xor
    # y = y_xor

    # DONUT
    N = 1000
    R_inner = 5
    R_outer = 10
    R1 = np.random.randn(N / 2) + R_inner
    theta = 2 * np.pi * np.random.random(N / 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    R2 = np.random.randn(N / 2) + R_outer
    theta = 2 * np.pi * np.random.random(N / 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    X = np.concatenate([X_inner, X_outer])
    y = np.array([0] * (N / 2) + [1] * (N / 2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # STANDARD ===============================
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # OUTPUT ==================================
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    decision_region_graph(X_train_std, y_train, X_test_std, y_test,forest)


if __name__ == '__main__':
    main()