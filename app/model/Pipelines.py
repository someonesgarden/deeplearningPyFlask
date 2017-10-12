#!/usr/bin/env python3
# _*_ coding:utf-8 _*_

import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy import interp
from sklearn.svm import SVC


class PipelinesClass(object):

    def __init__(self):
        self.df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
        self.X = self.df.loc[:, 2:].values
        self.y = self.df.loc[:, 1].values
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=1)
        self.kfold = StratifiedKFold(y=self.y_train, n_folds=3, random_state=1)

    def cross_validation(self, estimator, cv, n_jobs):
        scores = cross_val_score(estimator=estimator, X=self.X, y=self.y, cv=cv, n_jobs=n_jobs)
        print('CV accuracy scores: %s' % scores)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    def plot_validation_curve(self, estimator, cv):
        param_range = [0.001, 0.01,0.1,1.0, 10.0, 100.0]
        train_scores, test_scores = validation_curve(estimator=estimator, X=self.X_train, y=self.y_train,
                                                     param_name='clf__C', param_range=param_range, cv=cv)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(param_range, test_mean, color='green', marker='s', markersize=5, label='Validation Accuracy')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()
        plt.xscale('log')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim([0.8, 1.0])
        plt.show()

    def plot_learning_curve(self, estimator, cv, n_jobs):
        scores = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        X_train2 = self.X_train[:, [4,14]]
        fitted = estimator.fit(X_train2, self.y_train)
        y_pred2 = fitted.predict(self.X_test[:, [4, 14]])

        for k, (train, test) in enumerate(self.kfold):
            probas = estimator.fit(X_train2[train], self.y_train[train]).predict_proba(X_train2[test])
            score = estimator.score(X_train2[test], self.y_train[test])
            scores.append(score)
            print('Fold: %s, Class dist.:%s, Acc:%.3f' % (k + 1, np.bincount(self.y_train[train]), score))

            fpr, tpr, thresholds = roc_curve(self.y_train[test], probas[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (k+1, roc_auc))

        plt.plot([0, 1],
                 [0, 1],
                 linestyle='--',
                 color=(0.6, 0.6, 0.6),
                 label='random guessing')
        mean_tpr /= len(self.kfold)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.plot([0, 0, 1],
                 [0, 1, 1],
                 lw=2,
                 linestyle=':',
                 color='black',
                 label='perfect performance')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('Receiver Operator Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        print('ROC AUC: %.3f' % roc_auc_score(y_true=self.y_test, y_score=y_pred2))

        # print scores
        # print 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
        #
        # train_sizes, train_scores, test_scores = \
        #     learning_curve(estimator=estimator, X=self.X_train, y=self.y_train,
        #                    train_sizes=np.linspace(0.1, 1.0, 10), cv=cv, n_jobs=n_jobs)
        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        # test_mean = np.mean(test_scores, axis=1)
        # test_std = np.std(test_scores, axis=1)
        #
        # plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
        # plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        # plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation Accuracy')
        # plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        #
        # plt.grid()
        # plt.legend(loc='lower right')
        # plt.xlabel('Number of training samples')
        # plt.ylabel('Accuracy')
        #
        # plt.ylim([0.8, 1.0])
        # plt.show()

    def grid_search(self, estimator, cv, n_jobs, mode='grid'):
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
                        {'clf__C': param_range, 'clf__kernel': ['linear']},
                        {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}
                    ]
        param_distributions = {
            'clf__C': param_range,
            'clf__gamma': param_range,
            'clf__kernel': ['rbf'],
            'clf__kernel': ['linear']}

        if mode is 'grid':
            gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=n_jobs)

        else: #FAST!!
            gs = RandomizedSearchCV(estimator, param_distributions, cv=cv, n_iter=20, n_jobs=n_jobs, scoring="accuracy", verbose=2)

        gs = gs.fit(self.X_train,self.y_train)
        print(gs.best_score_)
        print(gs.best_params_)

        # Estimate the performance of the best selected model using "independent" test dataset
        clf = gs.best_estimator_
        clf.fit(self.X_train, self.y_train)
        print('Test accuracy:" %.3f' % clf.score(self.X_test, self.y_test))

        # cross_val_score
        scores = cross_val_score(gs, self.X, self.y, scoring='accuracy', cv=5)
        print('CV accuracy:" %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    def confusion_mat(self, estimator):
        estimator.fit(self.X_train, self.y_train)
        y_pred = estimator.predict(self.X_test)
        confmat = confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        print(confmat)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j],
                        va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.show()

    def f1_score(self,estimator):
        estimator.fit(self.X_train, self.y_train)
        y_pred = estimator.predict(self.X_test)
        precision = precision_score(y_true=self.y_test, y_pred=y_pred)
        recall = recall_score(y_true=self.y_test, y_pred=y_pred)
        f1 = f1_score(y_true=self.y_test, y_pred=y_pred)

        print("Presicion:%.3f, Recall: %.3f ,F1: %.3f" % (precision, recall, f1))

    def roc(self, estimator):
        X_train2 = self.X_train[:, [4,14]]
        cv = StratifiedKFold(self.y_train, n_folds=3, random_state=1)
        fig = plt.figure(figsize=(7,5))
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []




pipe = list()
pipe.append(('scl', StandardScaler()))
# pipe.append(('pca', PCA(n_components=2)))
pipe.append(('clf', LogisticRegression(penalty='l2', random_state=0)))

# processes = [
#         ('scl', StandardScaler()),
#         ('pca', PCA(n_components=2)),
#         ('clf', LogisticRegression(penalty='l2', random_state=0))
#     ]

pipe_lr = Pipeline(pipe)
pipe_svc = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', SVC(random_state=1))
    ])


pl = PipelinesClass()
pl.plot_learning_curve(pipe_lr, 10, 1)

# pl.cross_validation(pipe_lr, 10, 1)
# pl.plot_validation_curve(pipe_lr, 10)
# pl.grid_search(pipe_svc, 10, 1, mode='random')
# pl.confusion_mat(pipe_svc)
# pl.f1_score(pipe_svc)
