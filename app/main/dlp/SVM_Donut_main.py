#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

'''
kernel SVM (Support Vector Machine)
DONUT random distributions
'''

import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')

from sklearn import datasets as ds
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from app.lib.util import decision_region_graph

def main(C=10.0,gamma=2.45):
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
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
    svm = SVC(kernel='rbf', C=C, random_state=0, gamma=gamma)
    decision_region_graph(X_train_std, y_train, X_test_std, y_test,svm)


if __name__ == "__main__":
    main()