#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np


class ADAlineSGD(object):

    """
    ADAline_SGD (Adaptive Linear Neurons Stochastic Gradient Descent)
    ΔW = -η * ΔJ
    η(eta): (between 0.0 and 1.0)
    """

    def __init__(self, eta=0.01, n_iter=100, shuffle = True, random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, T):
        N, D = X.shape
        self._initialize_weights(D)
        self.cost_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, T = self._shuffle(X, T)

            cost = []
            for xi, target in zip(X, T):
                cost.append(self._update_weights(xi, target)[0])
            avg_cost = sum(cost)/len(T)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, T):
        #self.cost_ = []
        # 重みは最初帰化を行わない
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if T.ravel().shape[0] > 1:
            for xi, target in zip(X,T):
                self._update_weights(xi,target)
        else:
            self._update_weights(X, T)
        return self

    def _update_weights(self, X, T):
        output = self.net_input(X)
        errors = (T - output)
        self.w_[0] += self.eta * errors.sum()
        self.w_[1:] += self.eta * X.T.dot(errors)
        cost = 0.5 * errors**2
        # print(cost)
        return cost, errors

    """
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * xi.dot(error)
        cost = 0.5 * error**2
        print(cost)
        return cost, error
    """

    def _initialize_weights(self, D):
        # self.w = np.zeros(D+1)
        self.w_ = np.random.randn(D+1)
        self.w_initialized = True

    def net_input(self, X):
        return X.dot(self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

    def _shuffle(self, X, T):
        r = np.random.permutation(len(T))
        return X[r], T[r]
