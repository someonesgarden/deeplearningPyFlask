#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class TitleParam(object):

        def __init__(self, param):
            self.a0 = param['eta']
            self.a1 = param['epoch']
            self.a2 = param['fit_type']
            self.a3 = param['target_colname']
            self.a4 = param['target_col']
            self.a5 = param['x1_col']
            self.a6 = param['x2_col']
            self.a7 = param['url']
            self.a8 = param['sample_max']
            self.a9 = param['fit_name']
            self.a10= param['sample_scaling_title']


#X -- W1 -- Z -- W2 -- Y

def forward(X, W1, b1, W2, b2):
    # assume we will use tanh() on hidden
    # and softmax on output
    Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))
    activation = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-activation))
    return Y, Z


def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)


def derivative_w2(Z, T, Y):
    # Z is (N, M)
    return (T - Y).dot(Z)

def derivative_b2(T, Y):
    return (T - Y).sum()


def derivative_w1(X, Z, T, Y, W2):
    front = np.outer(T-Y, W2) * Z * (1 - Z)
    return front.T.dot(X).T


def derivative_b1(Z, T, Y, W2):
    front = np.outer(T-Y, W2) * Z * (1 - Z)
    return front.sum(axis=0)


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1.0 / (1.0 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

#cross_entropy in bad way
def cross_entropy(T, Y):
    E = 0
    for i in range(T.size):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

#cross entropy in good way
def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

def cost3(T, Y):
    tot = 0
    for n in range(len(T)):
        if T[n] == 1:
            tot += np.log(Y[n])
        else:
            tot += np.log(1 - Y[n])
    return tot


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y


def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) / K
    errors = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:")
    print(errors)
    return np.mean(errors)

def plt_decision_regions(X, T, classifier, resolution=0.02,par=None):
    markers = ('s', 'x', 'o', '^', 'v', '1', '2', '3', '4', '8', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'magenta', 'green', '#23ad11', '#2343fa', '#ac123a',
              '#e12acd', '#23ff1a', '#12a11f', '#ddde11', '#1a23a1', '#caca12', '#e1aa11', '#212aaa')
    cmap = ListedColormap(colors[:len(np.unique(T))])

    # 決定領域
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイント
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    if par is None:
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(T)):
            plt.scatter(x=X[T == cl, 0], y=X[T == cl, 1], alpha=0.98, c=cmap(idx), marker=markers[idx], label=cl)
    else:
        par.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        par.set_xlim(xx1.min(), xx1.max())
        par.set_ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(T)):
            par.scatter(x=X[T == cl, 0], y=X[T == cl, 1], alpha=0.98, c=cmap(idx), marker=markers[idx], label=cl)


def plot_decisionregions(X, X1, X2, Z, y, par=None, test_idx=None, param=None):

    #plt.clf()
    markers = ('s', 'x', 'o', '^', 'v','1','2','3','4','8','p','*','h','H','+','x','D','d')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','magenta','green','#23ad11','#2343fa','#ac123a',
              '#e12acd','#23ff1a','#12a11f','#ddde11','#1a23a1','#caca12','#e1aa11','#212aaa')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    if par is None:
        plt.contourf(X1, X2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.98, c=cmap(idx), marker=markers[idx], label=cl)
    else:
        par.contourf(X1, X2, Z, alpha=0.3, cmap=cmap)
        par.set_xlim(X1.min(), X1.max())
        par.set_ylim(X2.min(), X2.max())
        for idx, cl in enumerate(np.unique(y)):
            par.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.98, c=cmap(idx), marker=markers[idx], label=cl)

    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx,:], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='v', s=65, label='test set')

    if param is not None:
        plt.xlabel('xlabel:  {0}'.format(param['x1_colname']))
        plt.ylabel('ylabel:  {0}'.format(param['x2_colname']))

        title_txt = "{0.a9} fit={0.a2} T={0.a3}(col={0.a4}) [x1,x2]=[{0.a5},{0.a6}]" \
                    "eta={0.a0:.03f},epoch={0.a1} sams_num={0.a8} samp_scaln={0.a10}\nsrc='{0.a7}'"
        title_param = TitleParam(param)

        plt.title(title_txt.format(title_param), ha='center', fontsize=8, fontname='serif')
        plt.legend(loc='upper left')
        plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')


def Xstdardize(X):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std

def decision_region_graph(X_train_std, y_train, X_test_std, y_test, classifier):

    # FIT ====================================
    classifier.fit(X_train_std, y_train)

    # PREDICTION =============================
    y_pred = classifier.predict(X_test_std)

    # # ANALYSE ================================
    missed_samples = (y_test !=y_pred).sum()
    print("Classifier:", classifier)
    print("Misclassified : {0} ({1:.2f}%)".format(missed_samples,  (missed_samples*100/len(y_test))))
    print("Accuracy: {0:.2f}".format(accuracy_score(y_test,y_pred)))

    # =============================
    resolution = 0.02
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    x1_min, x1_max = X_combined_std[:, 0].min() - 1, X_combined_std[:, 0].max() + 1
    x2_min, x2_max = X_combined_std[:, 1].min() - 1, X_combined_std[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)

    plot_decisionregions(X_combined_std, X1, X2, Z, y_combined, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal weight [standardized]')
    plt.legend(loc='upper left')
    plt.show()