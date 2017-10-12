#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')
import numpy as np
import matplotlib.pyplot as plt
from app.lib.util import cost, cost3, forward, predict, derivative_w1,derivative_w2,derivative_b2,derivative_b1
# for binary classification! no softmax here

# def cost(T, Y):
#     tot = 0
#     for n in xrange(len(T)):
#         if T[n] == 1:
#             tot += np.log(Y[n])
#         else:
#             tot += np.log(1 - Y[n])
#     return tot


def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(3)
    W2 = np.random.randn(3)
    b2 = np.random.randn(1)
    LL = [] # keep track of likelihoods
    learning_rate = 0.0005
    regularization = 0.
    last_error_rate = None
    for i in range(100000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        if er != last_error_rate:
            last_error_rate = er
            print("error rate:", er)
            print("true:", Y)
            print("pred:", prediction)
            print("ll:", ll)
        if LL and ll > LL[-1]:
            print("early exit")
            break
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 10000 == 0:
            print(ll)
    print("final classification rate:", 1 - np.abs(prediction - Y).mean())
    plt.plot(LL)
    plt.show()


def test_donut():
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = [] # keep track of likelihoods
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None
    for i in range(16000):
        pY, Z = forward(X, W1, b1, W2, b2)
        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (derivative_w2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (derivative_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (derivative_w1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (derivative_b1(Z, Y, pY, W2) - regularization * b1)
        if i % 100 == 0:
            print("ll:", ll, "classification rate:", 1 - er)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    test_xor()
    test_donut()

    


