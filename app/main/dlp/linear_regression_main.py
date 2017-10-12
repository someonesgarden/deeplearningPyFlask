#!/usr/bin/env python3
# -*- coding: UTF-8 -*-# enable debugging

import sys
sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.model.LinearRegression import *
from app.model.LogisticRegression import *
from app.lib.util import plot_decisionregions
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from app.model.ADAlineSGD import *
from sklearn.svm import SVC


def main(learning_rate=0.1, EPOCH=10, source='/Users/user/PycharmProjects/deeplearningFlask/static/uploaded/iris.data.csv'):
    #source= 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print("Linear_Regression")
    df = pd.read_csv(source, header=None)
    y = df.iloc[0:100,4].values

    T = np.where(y == 'Iris-setosa', 1, -1)
    X = df.iloc[0:100, [0,2]].values

    plt.clf()
    plt.scatter(X[:50,0], X[:50,1],color='red', marker='o')
    plt.scatter(X[50:,0], X[50:,1],color='blue',marker='x')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    #plt.show()


    ### FIT
    fit_type = 2
    fit_name = "Linear Regression"

    if fit_type ==1:
        fit_name="ADAlineSGD"
        ppn = ADAlineSGD(eta=0.1, n_iter=20, random_state=1)

    elif fit_type == 2:
        fit_name="Support Vector Machine(RBF)"
        ppn = SVC(kernel='rbf', C=0.1, random_state=0, gamma=0.15)

    else:
        fit_name="Linear Regression"
        ppn = LinearRegression(learning_rate=learning_rate, EPOCH=EPOCH)

    ppn.partial_fit(X, T)
    #plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_)
    #plt.show()

    ###
    resolution = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = ppn.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)
    print(Z.shape)

    plot_decisionregions(X, X1, X2, Z, y)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title("Liear Regression(eta={0:.03f},epoch={1})".format(learning_rate,EPOCH), fontsize=25, fontname='serif')  # タイトル
    plt.legend(loc = 'upper left')
    #plt.show()
    plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')

def predict_with_param(param):
    print("Linear_Regression")
    print(param)
    sample_max = int(param['sample_max'])
    source = param['url']  #if param['url'] is not None else '/Users/user/PycharmProjects/DeepLearningPy/static/uploaded/iris.data.csv'
    learning_rate = float(param['learning_rate'])
    EPOCH = int(param['epoch'])
    random_state = int(param['random_state'])
    svc_gamma = float(param['svc_gamma'])
    svc_kernel = str(param['svc_kernel'])
    fit_type = int(param['fit_type']['type'])
    learn_c = float(param['learn_c'])
    x1_col, x2_col = int(param['x1']), int(param['x2'])
    target_col = int(param['target'])
    header_is = 0 if param['header_is'] else None

    df = pd.read_csv(source, header=header_is)

    ## Null Treatment
    s = 0
    for i, d in enumerate(df.isnull().sum()):
        s += d
    if s > 0:
        if param['null_treatment']:
            df = df.dropna()
        else:
            imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imr.fit(df)
            columns = df.columns
            df = pd.DataFrame(imr.transform(df.values), columns=columns)

    sample_max = sample_max if (sample_max < df.shape[0] and sample_max != 0) else df.shape[0]
    y = df.iloc[0:sample_max, target_col].values
    X = df.iloc[0:sample_max, [x1_col, x2_col]].values

    print(X)

    T = y
    if param['target_modify'] == 0:
        # Leave as it is.
        print("target_mdify:0")

    elif param['target_modify'] == 1:
        # map to [0, ~]
        tmp = np.unique(y)
        T = np.where(y == tmp[0], 1, -1)
        print("target_modify:1")
    elif param['target_modify'] == 2:
        print("target_modify:2")
        class_le = LabelEncoder()
        T = class_le.fit_transform(y)

    #############
    plt.clf()
    ### FIT
    fit_name = "Linear Regression"

    if fit_type == 1:
        fit_name = "ADAlineSGD"
        ppn = ADAlineSGD(learning_rate=learning_rate, EPOCH=EPOCH, random_state=random_state)

    elif fit_type == 2:
        fit_name = "Support Vector Machine(RBF)"
        ppn = SVC(kernel=svc_kernel, C=learn_c, random_state=random_state, gamma=svc_gamma)

    elif fit_type == 3:
        fit_name = "Logistic Regression"
        ppn = LogisticRegression(learning_rate=learning_rate, EPOCH=EPOCH)

    else:
        fit_name = "Linear Regression"
        #learning_rate
        #epoch
        ppn = LinearRegression(learning_rate=learning_rate, EPOCH=EPOCH)

    ppn.fit(X, T)
    # plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_)
    # plt.show()


    # plt.scatter(X[:50,0], X[:50,1],color='red', marker='o')
    # plt.scatter(X[50:,0], X[50:,1],color='blue',marker='x')
    #plt.scatter(X[:xmax, 0], X[:xmax, 1], color='red', marker='^')
    #plt.show()
    #plt.show()

    ###
    resolution = 0.02
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print("x1_min:{0:.02f}, x1_max:{1:.02f}".format(x1_min,x1_max))
    X1, X2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = ppn.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)
    print("Z.shape")
    print(Z.shape)

    plot_decisionregions(X, X1, X2, Z, y)
    plt.xlabel('xlabel:  {0}'.format(param['x1_colname']))
    plt.ylabel('ylabel:  {0}'.format(param['x2_colname']))
    plt.title("fit={2} T={3}(col={4}) [x1,x2]=[{5},{6}] eta={0:.03f},epoch={1} samples={8}\nsrc='{7}'".format(learning_rate, EPOCH, fit_name, param['target_colname'],target_col, x1_col, x2_col, source,sample_max), ha='center', fontsize=8, fontname='serif')  # タイトル
    plt.legend(loc = 'upper left')
    #plt.show()
    plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')

if __name__ == '__main__':
    print("goto main")
    main()