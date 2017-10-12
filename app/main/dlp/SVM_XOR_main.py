#!/usr/bin/env python3
# _*_ coding:utf-8 _*_


'''
kernel SVM (Support Vector Machine)
XOR random distributions
'''

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


def main(C=10.0,gamma=2.45):
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    X = X_xor
    y = y_xor

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