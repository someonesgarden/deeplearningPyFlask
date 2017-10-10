from scipy.misc import comb
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from model.MajorityVoteClassifier import *
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import operator
from itertools import product


class Ensembles(object):

    def __init__(self):

        self.colors = ['black', 'orange','blue','green']
        self.linestyles =[':', '--', '-.', '-']
        self.sc = StandardScaler()
        self.mv_clf=None
        self.X_train =None
        self.y_train =None

    def ensemble_error(self, n_classifier, error):
        k_start = int(math.ceil(n_classifier / 2.0))
        probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k) for k in range(k_start, n_classifier + 1)]
        return sum(probs)

    def ensemble_plot(self):
        error_range = np.arange(0.0, 1.01, 0.01)
        ens_errors = [self.ensemble_error(n_classifier=11,error=error) for error in error_range]

        plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
        plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)

        plt.xlabel('Base error')
        plt.ylabel('Base/Ensemble error')
        plt.legend(loc='upper left')
        plt.grid()
        plt.show()

    def roc_auc_func(self, mode='density'):
        le = LabelEncoder()
        iris = datasets.load_iris()
        X, y = iris.data[50:, [1, 2]], iris.target[50:]
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
        X_train_std = self.sc.fit_transform(X_train)
        self.X_train = X_train_std
        self.y_train = y_train

        clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)
        clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
        clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
        pipe1 = Pipeline([
            ['sc', StandardScaler()],
            ['clf', clf1]])
        pipe2 = Pipeline([
            ['sc',StandardScaler()],
            ['clf', clf2]])
        pipe3 = Pipeline([
            ['sc', StandardScaler()],
            ['clf', clf3]])

        clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
        print('10-fold cross validation: \n')
        print('ROC AUC : Receiver Operator Characteristic Area Under the Curve.\n')
        for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
            scores = cross_val_score(estimator=clf,
                                     X=X_train,
                                     y=y_train,
                                     cv=10,
                                     scoring='roc_auc')
            scores_t = cross_val_score(estimator=clf,
                                       X=X_test,
                                       y=y_test,
                                       cv=10,
                                       scoring='roc_auc')
            print("training ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
            print("test ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores_t.mean(), scores_t.std(), label))

        self.mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

        clf_labels += ['Majority Voting']
        all_clf = [pipe1, clf2, pipe3, self.mv_clf]
        for clf, label in zip(all_clf, clf_labels):
            scores = cross_val_score(estimator=clf,
                                     X=X_train, y=y_train,
                                     cv=10,
                                     scoring='roc_auc')
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

        if mode == 'density':
            plt.clf()
            x_min = X_train_std[:, 0].min() - 1
            x_max = X_train_std[:, 0].max() + 1
            y_min = X_train_std[:, 1].min() - 1
            y_max = X_train_std[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

            f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(7, 5))
            for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
                clf.fit(X_train_std, y_train)
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
                axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                              X_train_std[y_train==0, 1],
                                              c='blue',
                                              marker='^',
                                              s=50)
                axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                              X_train_std[y_train==1, 1],
                                              c='red',
                                              marker='o',
                                              s=50)
                axarr[idx[0], idx[1]].set_title(tt)

            plt.text(-3.5,-4.5, s='Sepal width [ standardized]',
                     ha='center', va='center', fontsize=12)
            plt.text(-10.5, 4.5, s='Petal length [standard]',
                     ha='center', va='center', fontsize=12, rotation=90)
            plt.show()

        else:
            plt.clf()
            for clf, label, clr, ls in zip(all_clf, clf_labels, self.colors, self.linestyles):
                # assuming the label of the positive class is 1
                y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
                roc_auc = auc(x=fpr, y=tpr)
                plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))

            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.grid()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()