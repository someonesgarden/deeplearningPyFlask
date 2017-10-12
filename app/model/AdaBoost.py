#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

url0 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'


class MyAdaBoost(object):

    def __init__(self, url=url0):
        df = pd.read_csv(url, header=None)
        df.columns = ['Class label',
                           'Alcohol',
                           'Malic alid',
                           'Ash',
                           'Alcalinity of ash',
                           'Magnesium',
                           'Total phenols',
                           'Flavonoids',
                           'Nonflavonoid phenols',
                           'Proanthocyanins',
                           'Color intensity',
                           'Hue',
                           'OD280/OD315 of diluted wines',
                           'Proline']

        df = df[df['Class label'] != 1]
        self.y = df['Class label'].values
        self.X = df[['Alcohol', 'Hue']].values

        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.40, random_state=1)

        self.tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        self.ada = AdaBoostClassifier(base_estimator=self.tree, n_estimators=500, learning_rate=0.1, random_state=0)


    def check_accuracies(self):

        # Fit

        tree = self.tree.fit(self.X_train, self.y_train)
        y_train_pred = tree.predict(self.X_train)
        y_test_pred = tree.predict(self.X_test)
        tree_train = accuracy_score(self.y_train, y_train_pred)
        tree_test = accuracy_score(self.y_test, y_test_pred)
        print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

        ada = self.ada.fit(self.X_train, self.y_train)
        y_train_pred = ada.predict(self.X_train)
        y_test_pred = ada.predict(self.X_test)
        ada_train = accuracy_score(self.y_train, y_train_pred)
        ada_test = accuracy_score(self.y_test, y_test_pred)
        print('Adaptive Boost train/test accuracies  %.3f/%.3f' % (ada_train, ada_test))

    def decision_region(self):

        x_min = self.X_train[:, 0].min() - 1
        x_max = self.X_train[:, 0].max() + 1
        y_min = self.X_train[:, 1].min() - 1
        y_max = self.X_train[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8,3))
        for idx, clf, tt in zip([0,1], [self.tree, self.ada], ['Decision Tree', 'AdaBoost']):

            clf.fit(self.X_train,self. y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            axarr[idx].contourf(xx, yy, Z, alpha=0.3)
            axarr[idx].scatter(self.X_train[self.y_train==0, 0],
                               self.X_train[self.y_train==0, 1],
                               c='blue',  marker='^')
            axarr[idx].scatter(self.X_train[self.y_train==1, 0],
                               self.X_train[self.y_train==1, 1],
                               c='red', marker='o')
            axarr[idx].set_title(tt)
        axarr[0].set_ylabel('Alcohol', fontsize=12)

        plt.text(10.2, -1.2, s='Hue', ha='center', va='center', fontsize=12)
        plt.show()


if __name__ == "__main__":
    mada = MyAdaBoost()
    mada.check_accuracies()
    mada.decision_region()