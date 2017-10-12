#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np

class Perceptron(object):
    """
    パーセプトロンの分類器

    パラメータ
    -------------
    eta : float      学習率(0.0<eta<=1.0)
    n_iter : int     トレーニングデータのトレーニング回数

    属性
    --------------
    w_ : 1d ndarray  適合後の重み
    errors_ : list   各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit_vector(self, X,T):
        # パーセプトロンの場合、「先に個々の重み」を計算して重みを更新する必要があるので、
        # ベクターでまとめて演算は難しい。（この関数は不正確）
        N,D = X.shape
        self.errors_ = []
        self._initialize_weights(D)

        for _ in range(self.n_iter):
            update = self.eta * (T - self.predict(X))
            output= np.where(update != 0.0, 1,0)
            errors_ = output.sum()
            self.w_[0] += update.sum()
            self.w_[1:] += X.T.dot(update)
            self.errors_.append(errors_)

        return self


    def fit(self, X, T):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):

            errors=0
            i = 0
            print(i)

            for xi, target in zip(X, T):
                #各サンプルで重みを更新
                #総入力からクラスラベルを予測
                #誤差*学習率＝update
                update = self.eta * (target - self.predict(xi))
                errors += int(update != 0.0)

                self.w_[1:] += update * xi
                self.w_[0]  += update

            self.errors_.append(errors)
            i += 1

        return self


    def _initialize_weights(self, D):
        self.w_ = np.zeros(D+1)
        self.w_initialized = True

    def net_input(self, X):
        return X.dot(self.w_[1:]) + self.w_[0]
        #return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """クラスラベルを返す"""
        return np.where(self.net_input(X) >=0.0, 1, -1)


