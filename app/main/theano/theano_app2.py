#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import theano
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1, 1.4, 1.5]])
w = np.array([0.0, 0.2, 0.4])


def net_input(X, w):
    z = X.dot(w)
    return z


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_activatioin(X, w):
    z = net_input(X, w)
    return logistic(z)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def softmax_activation(X, w):
    z = net_input(X, w)
    return softmax(z)

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


def main():
    print('P(y=1|x) = %.3f' % logistic_activatioin(X, w)[0])

    W = np.array([[1.1, 1.2, 1.3, 0.5],
                  [0.1, 0.2, 0.4, 0.1],
                  [0.2, 0.5, 2.1, 1.9]])
    A = np.array([[1.0],
                  [0.1],
                  [0.3],
                  [0.7]])
    Z = W.dot(A)
    y_probas = logistic(Z)
    y_probas2 = softmax(Z)
    print('Probabilities logistic:\n', y_probas)

    print('Probabilities softmax:\n', y_probas2)


if __name__ == '__main__':
    main()