#!/usr/bin/env python3
# -*- coding: UTF-8 -*-# enable debugging

import sys
sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")

try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

import re
import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np
from pandas import Series

#url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
http_regex = re.compile("http[s]?")
local_regex = re.compile("static\/uploaded?")

param0 = {
    'shape': None,
    'head': None,
    'url': '/Users/user/PycharmProjects/DeepLearningPy/static/uploaded/iris.data.csv',
    'status': True,
    'message': 'Data Preprocessing..\n',
    'header_is': 0,
    'null_treatment': 0
}

class DataPreProcess(object):

    def __init__(self):

        self.param = {
            'shape': None,
            'head': None,
            'column':None,
            'url': '',
            'status': True,
            'message': '',
            'header_is': False,
            'null_treatment': False,
            'null_num':0
        }

    def validate_url(self, param=param0):
        print(param)
        self.df = pd.DataFrame()

        self.param['url'] = param['url']
        self.param['status'] = param['status']
        self.param['message'] = param['message']
        self.param['header_is'] = param['header_is']
        self.param['null_treatment'] = param['null_treatment']

        # 正しくhttpで始まっているかどうか
        if http_regex.search(self.param['url']) is not None:
            try:
                urllib2.urlopen(self.param['url'], timeout=10)
            except:
                pass
            # except urllib2.URLError, e:
            #
            #     # 接続テスト失敗
            #     print e
            #     self.param['status'] = False
            #     self.param['message'] += str(e)

        # URLも存在し、ローカルにもありそうかどうか
        if self.param['status'] or local_regex.search(self.param['url']) is not None:

            header_is = 0 if self.param['header_is'] else None
            print("df = pd.read_csv(self.param['url'], header=header_is)")
            self.df = pd.read_csv(self.param['url'], header=header_is)

            # DataFrameにNullがあるかどうかを判別
            s = 0
            for i, d in enumerate(self.df.isnull().sum()):
                s += d

            if s > 0:
                print("null_s is %d" % s)
                self.param['null_num']=s
                # 要素に一つでもNullがある場合
                if self.param['null_treatment']:
                    #NULLはDeleteして処理する場合
                    self.df = self.df.dropna()
                    self.param['message'] += "NULL Deleted.\n"
                else:
                    #NULLは平均をImpute して処理する場合
                    print("impute with mean values")
                    try:
                        imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
                        imr.fit(self.df)
                    except:
                        pass
                    # except UnboundLocalError, e:
                    #     pass
                    columns = self.df.columns
                    self.df = pd.DataFrame(imr.transform(self.df.values), columns=columns)
                    self.param['message'] += "NULL Imputed.\n"
            else:
                print("null is 0")

            self.param['shape'] = self.df.shape
            self.param['column'] = Series(self.df.columns).to_json()
            self.df = self.df.iloc[np.unique(np.random.randint(0, self.df.shape[0]-1, 8))]
            self.param['head'] = self.df.T.to_json()
            self.param['head_t'] = self.df.to_json()

        else:
            self.param['status'] = False
            self.param['message'] += "invalid data / data not found\n"

        return self.param

    def target_modify(self, param=param0):
        param = self.validate_url(param);
        print(param)

#
#
# from sklearn.cross_validation import train_test_split
# X,y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
#
#
# #normalization
# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# X_train_norm=mms.fit_transform(X_train)
# X_test_norm=mms.fit_transform(X_test)
#
# #standardization : Z-value
# from sklearn.preprocessing import StandardScaler
# stdsc = StandardScaler()
# X_train_std = stdsc.fit_transform(X_train)
# X_test_std = stdsc.fit_transform(X_test)
#
# from lib.sbs import *
# from sklearn.neighbors import KNeighborsClassifier
# import matplotlib.pyplot as plt
#
# knn = KNeighborsClassifier(n_neighbors=2)
# sbs = SBS(knn,k_features=1)
# sbs.fit(X_train_std, y_train)
#
# k_feat = [len(k) for k in sbs.subsets_]
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.1])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.show()
#
# k5 = list(sbs.subsets_[8])
# print ("k5:")
# print k5
# print (df_wine.columns[1:][k5])
# knn.fit(X_train_std[:, k5], y_train)
# print('Training accurary:', knn.score(X_train_std[:,k5], y_train))
# print('Test accuracy:', knn.score(X_test_std[:,k5], y_test))ｄf

