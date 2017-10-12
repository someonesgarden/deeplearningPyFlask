#!/usr/bin/env python3
# _*_ coding:utf-8 _*_


import sys
sys.path.append('/Users/user/PycharmProjects/deeplearningFlask')
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np

csv_data='''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,
'''
#csv_data = unicode(csv_data) # only for Python 2.7
df = pd.read_csv(StringIO(csv_data))

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
"""
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
"""

cat_data = '''
green,M,10.1,class1
red,L,13.5,class2
blue,XL,15.3,class1
brown,M,6.3,class3
silver,L,3.4,class1
'''

#cat_data = unicode(cat_data)
df2 = pd.read_csv(StringIO(cat_data), index_col=None)
df2.columns=['color','size','price','classlabel']
size_mapping = { "XL":3, "L":2, "M":1}
df2['size'] = df2['size'].map(size_mapping)
class_mapping={label:idx for label,idx in enumerate(np.unique(df2['classlabel']))}

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = [
    'Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids','Nonflavonoid phoenols', 'Preanthocyanins','Color intensity', 'Hue',
    'OD280/OD315', 'Proline'
]

from sklearn.cross_validation import train_test_split
X,y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)


#normalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm=mms.fit_transform(X_train)
X_test_norm=mms.fit_transform(X_test)

#standardization : Z-value
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

from app.lib.sbs import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn,k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print ("k5:")
print (k5)
print (df_wine.columns[1:][k5])
knn.fit(X_train_std[:, k5], y_train)
print('Training accurary:', knn.score(X_train_std[:,k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:,k5], y_test))