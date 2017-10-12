#!/usr/bin/env python3
# -*- coding: UTF-8 -*-# enable debugging

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.LinearRegression import *
from lib.LogisticRegression import LogisticRegression as logReg
from lib.util import plot_decisionregions
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.lda import LDA
from lib.ADAlineSGD import *
from sklearn.svm import SVC

sys.path.append("/Users/user/PycharmProjects/deeplearningFlask")
url0 = '/Users/user/PycharmProjects/DeepLearningPy/static/uploaded/iris.data.csv'


class FitParam(object):

    def __init__(self, param):

        model = param['model']
        query = param['query']
        x1, x2, t = model['x1'], model['x2'], model['t']

        self.param = {
            "eta": float(model['eta']),
            "epoch": int(model['epoch']),
            "sample_max": int(model['sample_max']),
            "random_state": int(model['random_state']),
            "gamma": float(model['gamma']),
            "kernel": str(model['kernel']),
            "learn_penalty": model['penalty'],
            "learn_c": float(model['c']),
            "sample_scaling": model['sample_scaling'],
            "pre_analysis": model['pre_analysis'],
            "test_size": float(model['test_size']),

            "target_col": int(t['selected']),
            "target_colname": t['colname'],
            "target_modify": t['modify'],
            "x1_col": int(x1['selected']),
            "x1_colname": x1['colname'],
            "x2_col": int(x2['selected']),
            "x2_colname": x2['colname'],

            "url": query['url'],
            "header_is": 0 if query['header_is'] else None,
            "null_treatment": query['null_del_is'],
            "check_fit_action": query['action'],
            "fit_type": int(query['algorithm_selected']['type'])
        }
        print("FitParam:__init__:param")
        print(self.param)


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
            self.a10 = param['sample_scaling_title']


class Fitting(object):

    def __init__(self, param):
        self.param = FitParam(param).param
        self.df = pd.DataFrame()

    def generate_xy(self, param):

        self.df = pd.read_csv(param['url'], header=param['header_is'])

        s = 0
        for i, d in enumerate(self.df.isnull().sum()):
            s += d
        if s > 0:
            if param['null_treatment']:
                self.df = self.df.dropna()
            else:
                imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
                imr.fit(self.df)
                columns = self.df.columns
                self.df = pd.DataFrame(imr.transform(self.df.values), columns=columns)

        sam_max = param['sample_max'] if (param['sample_max'] < self.df.shape[0] and param['sample_max'] is not 0) else self.df.shape[0]
        T = self.df.iloc[0:sam_max, param['target_col']].values
        X = self.df.iloc[0:sam_max, [param['x1_col'], param['x2_col']]].values

        X_colrange = np.arange(0, self.df.shape[1])
        X_colrange = np.delete(X_colrange, [param['target_col']], None)
        Xall = self.df.iloc[0:sam_max, X_colrange].values
        print(Xall)

        if param['target_modify'] == 0:
            # Leave as it is.
            print("target_mdify:0")

        elif param['target_modify'] == 1:
            # map to [0, ~]
            tmp = np.unique(T)
            T = np.where(T == tmp[0], 1, -1)
            print("target_modify:")
            print("1={0}".format(tmp[0]))
        elif param['target_modify'] == 2:
            print("target_modify:2")
            class_le = LabelEncoder()
            print("Before LabelEncode")
            print(np.unique(T))
            T = class_le.fit_transform(T)
            print("After LabelEncode")
            print(np.unique(T))
        return X, T, Xall

    def pca_fit(self, Xall_train):

        stdsc = StandardScaler()
        Xall_train = stdsc.fit_transform(Xall_train)
        cov_mat = np.cov(Xall_train.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        # Feature Transformation
        # CHOOSE ONLY TWO EIGEN VALUES!!! (~60%)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range((len(eigen_vals)))]
        eigen_pairs.sort(reverse=True)
        w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
        Xall_train_pca = Xall_train.dot(w)
        Xall_train_pca_vstack = np.vstack([v for v in Xall_train_pca])
        return Xall_train_pca_vstack

    def fit_data_with_model(self, param, X, T):
        print(param)
        # FIT
        fit_type = param['fit_type']
        if fit_type == 1:
            self.param['fit_name'] = "ADAlineSGD"
            ppn = ADAlineSGD(learning_rate=param['eta'], EPOCH=param['epoch'], random_state=param['random_state'])

        elif fit_type == 2:
            self.param['fit_name'] = "Support Vector Machine(RBF)"
            ppn = SVC(kernel=param['kernel'], C=param['learn_c'], random_state=param['random_state'],
                      gamma=param['gamma'])
        elif fit_type == 3:
            self.param['fit_name'] = "Logistic Regression"

            ppn = LogisticRegression(penalty=param['learn_penalty'], C=param['learn_c'])
            #ppn = LogisticRegression()
            #ppn = logReg(learning_rate=param['eta'], EPOCH=param['epoch'])

        elif fit_type == 4:
            self.param['fit_name'] = "Decision Tree"
            ppn = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

        elif fit_type == 5:
            self.param['fit_name'] = "K-Nearest Neighbors"
            ppn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

        else:
            self.param['fit_name'] = "Linear Regression"
            # learning_rate
            # epoch
            ppn = LinearRegression(learning_rate=param['eta'], EPOCH=param['epoch'])
        ppn.fit(X, T)
        try:
            print("Training Accuracy:", ppn.score(X, T))
        except:
            pass
        # except AttributeError, e:
        #     print e
        #     pass


        ###
        resolution = 0.02
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        print("x1_min:{0:.02f}, x1_max:{1:.02f}".format(x1_min, x1_max))
        X1, X2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        Z = ppn.predict(np.array([X1.ravel(), X2.ravel()]).T)
        Z = Z.reshape(X1.shape)

        return X1, X2, Z

    def predict_with_param(self):

        # Data Generate
        X, T, Xall = self.generate_xy(self.param)
        X_train, X_test, T_train, T_test, Xall_train, Xall_test = train_test_split(X, T, Xall, test_size=self.param['test_size'], random_state=self.param['random_state'])

        self.param['sample_scaling_title'] = "Non"
        if self.param['sample_scaling'] is 1:
            self.param['sample_scaling_title'] = "Normalize"
            # Normalize
            mms = MinMaxScaler()
            X_train = mms.fit_transform(X_train)
            X_test = mms.fit_transform(X_test)

        elif self.param['sample_scaling'] is 2:
            self.param['sample_scaling_title']="Standardize"
            # Standardize
            stdsc = StandardScaler()
            X_train = stdsc.fit_transform(X_train)
            X_test = stdsc.fit_transform(X_test)

        if int(self.param['check_fit_action']) is 0:
            print(" int(self.param['check_fit_action']) == 0")
            # Plot Decision Regions

            if self.param['pre_analysis'] is 1:
                # PCA
                X_train = self.pca_fit(Xall_train)
            elif self.param['pre_analysis'] is 2:
                print("LDA")
                lda = LDA(n_components=2)
                X_train_lda = lda.fit_transform(Xall_train, T_train)
                X_train = np.vstack([v for v in X_train_lda])

                # LDA
            elif self.param['pre_analysis'] is 3:
                # Kernel PCA
                kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
                X_train_pca = kpca.fit_transform(Xall_train)
                X_train = np.vstack([v for v in X_train_pca])

            X1, X2, Z = self.fit_data_with_model(self.param, X_train, T_train)
            plot_decisionregions(X_train, X1, X2, Z, T_train, param=self.param)

        elif int(self.param['check_fit_action']) == 1:
            # Weight Coefficient
            plt.clf()
            fig = plt.figure()
            ax = plt.subplot(111)
            colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan', 'magenta', 'green', '#23ad11', '#2343fa', '#ac123a',
                      '#e12acd', '#23ff1a', '#12a11f', '#ddde11', '#1a23a1', '#caca12', '#e1aa11', '#212aaa']
            weights, pars =[], []

            for c in np.arange(-4, 6):
                lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
                lr.fit(Xall_train, T_train)
                weights.append(lr.coef_[0])
                pars.append(10**c)
            weights = np.array(weights)
            print(weights)
            for column, color in zip(range(weights.shape[1]), colors):
                plt.plot(pars, weights[:,column], label=self.df.columns[column+1], color=color)

            plt.axhline(0, color='black', linestyle='--',linewidth=2)
            plt.xlim([10**(-5),10**5])
            plt.ylabel('weight coefficient')
            plt.xlabel('C')
            plt.xscale('log')
            plt.legend(loc='upper left')
            ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03), ncol=1, fancybox=True)
            # plt.show()
            plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')

        elif int(self.param['check_fit_action']) == 2:
            # Feature Importance by Random Forest
            print("else")
            plt.clf()
            feat_labels = self.df.columns[1:]
            forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
            stdsc = StandardScaler()
            Xall_train = stdsc.fit_transform(Xall_train)
            forest.fit(Xall_train, T_train)

            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]

            for f in range(Xall_train.shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
            plt.title('Feature Importances for {0}'.format(self.param['target_colname']))
            plt.bar(range(Xall_train.shape[1]),
                    importances[indices],
                    color='lightblue',
                    align='center')
            plt.xticks(range(Xall_train.shape[1]),
                       feat_labels, rotation=90)
            plt.xlim([-1, Xall_train.shape[1]])
            plt.tight_layout()
            plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')

        elif int(self.param['check_fit_action']) == 3:
            plt.clf()
            # Principal Component Analysis (PCA)
            stdsc = StandardScaler()
            Xall_train = stdsc.fit_transform(Xall_train)
            X_test = stdsc.fit_transform(X_test)
            cov_mat = np.cov(Xall_train.T)
            eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

            tot = sum(eigen_vals)
            var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
            cum_var_exp = np.cumsum(var_exp)

            plt.bar(range(1,len(cum_var_exp)+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
            plt.step(range(1,len(cum_var_exp)+1), cum_var_exp, where='mid', label='cumulative explained variance')
            plt.ylabel('Explained variance ratio')
            plt.xlabel('Principal components')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('/Users/user/PycharmProjects/DeepLearningPy/static/out/linear_regression1.png')

            # Feature Transformation
            # CHOOSE ONLY TWO EIGEN VALUES!!! (~60%)
            eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range((len(eigen_vals)))]
            eigen_pairs.sort(reverse=True)
            w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
            Xall_train_pca = Xall_train.dot(w)









