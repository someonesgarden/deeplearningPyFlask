#!/user/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np


class ADAlineGD(object):

    """
    ADALINE_GD (Adaptive Linear Neurons : Gradient Descent)

    params
    -------------
    eta     : float   学習率(0.0<eta<=1.0)
    n_iter  : int     トレーニングデータのトレーニング回数

    attributes
    --------------
    w_      : 1d ndarray  適合後の重み
    errors_ : list        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, T):
        N,D = X.shape
        self.errors_ = []
        self._initialize_weights(D)
        self.cost_ = []
        for _ in xrange(self.n_iter):
            #活性化関数の出力： Φ(wTx) = wTx
            output = self.net_input(X)
            # 誤差
            errors = (T - output)
            # w の更新
            self.w[0]  += self.eta * errors.sum()
            self.w[1:] += self.eta * X.T.dot(errors)
            #コスト関数
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            #self.errors_.append(np.abs(errors).sum())
        return self

    def _initialize_weights(self, D):
        self.w = np.zeros(D+1)
        self.w_initialized = True

    def net_input(self, X):
        # 総入力
        return X.dot(self.w[1:]) + self.w[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):

        return np.where(self.activation(X) >= 0.0, 1, -1)

