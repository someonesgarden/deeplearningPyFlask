#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import pyprind
import pandas as pd
import numpy as np
import os
import os.path
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import re
import math

path2proj = "/Users/user/PycharmProjects/deeplearningFlask"
sys.path.append(path2proj)
stop = stopwords.words('english')

dest = os.path.join('app/movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

# Sentiment Analysis = Opinion Mining
# NLP = Natural Language Processing


class SentimentAnalysis(object):

    def __init__(self):
        self.df = pd.DataFrame()

        if not os.path.exists(path2proj+'/data/movie_data.csv'):
            self.load_csv_to_local()

    @staticmethod
    def load_csv_to_local():
        pbar = pyprind.ProgBar(50000)
        labels = {'pos': 1, 'neg': 0}
        df = pd.DataFrame()

        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = './data/aclImdb/%s/%s' % (s, l)
                for file_ in os.listdir(path):
                    with open(os.path.join(path, file_), 'r') as infile:
                        txt = infile.read()
                        df = df.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()

        np.random.seed(0)
        df.columns = ['review', 'sentiment']
        df = df.reindex(np.random.permutation(df.index))
        df.to_csv(path2proj+'/data/movie_data.csv', index=False)

    @staticmethod
    def preprocessor(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text)
        text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
        return text

    def main(self):
        if not os.path.exists(path2proj + '/data/movie_data.csv'):
            self.load_csv_to_local()
        self.df = pd.read_csv(path2proj+'/data/movie_data.csv')
        print self.df.head(3)

        count = CountVectorizer(ngram_range=(1,1))
        tfidf = TfidfTransformer()

        docs = np.array([
            'The sun is shining',
            'The weather is sweet',
            'The sun is shining and the weather is sweet'
        ])
        bag = count.fit_transform(docs)

        print count.vocabulary_
        print bag.toarray()

        np.set_printoptions(precision=2)
        print "tfidf.fit_transform(count.fit_transform(docs).toarray())"
        print tfidf.fit_transform(count.fit_transform(docs)).toarray()

        self.df['review'] = self.df['review'].apply(self.preprocessor)

        # test = [w for w in self.tokenizer_port('a runner likes running and runs a lot')[-10:] if w not in self.stop]
        # print test

        print "df:shape"
        print self.df.shape

        self.training_logistic_reg(self.df)

    def main_sgd(self):
        if not os.path.exists(path2proj + '/data/movie_data.csv'):
            self.load_csv_to_local()
        self.df = pd.read_csv(path2proj + '/data/movie_data.csv')
        print self.df.head(3)

        vect = HashingVectorizer(decode_error='ignore',
                                 n_features=2**21,
                                 preprocessor=None,
                                 tokenizer=tokenizer)
        clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
        doc_stream = stream_docs(path=path2proj+'/data/movie_data.csv')
        pbar = pyprind.ProgBar(45)
        classes = np.array([0, 1])
        for _ in range(45):
            X_train, y_train = get_minibatch(doc_stream, size=1000)
            if not X_train:
                break
            X_train = vect.transform(X_train)
            clf.partial_fit(X_train, y_train, classes=classes)
            pbar.update()

        X_test, y_test = get_minibatch(doc_stream, size=5000)
        X_test = vect.transform(X_test)
        print 'Accuracy: %.3f' % clf.score(X_test, y_test)
        pickle.dump(stop,
                    open(os.path.join(dest, 'stopwords.pkl'), 'wb')
                    )
        pickle.dump(clf,
                    open(os.path.join(dest,'classifier.pkl'), 'wb')
                    )

    def training_logistic_reg(self, df):

        self.X_train = df.loc[:25000, 'review'].values
        self.y_train = df.loc[:25000, 'sentiment'].values

        self.X_test = df.loc[25000:, 'review'].values
        self.y_test = df.loc[25000:, 'sentiment'].values

        tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
        param_grid = [
            {
                'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]
            },
            {
                'vect__ngram_range': [(1, 1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer, tokenizer_porter],
                'vect__use_idf': [False],
                'vect__norm':[None],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]
            }
        ]

        lr_tfidf = Pipeline([
            ('vect', tfidf),
            ('clf', LogisticRegression(random_state=0))
        ])

        gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=2)

        gs_lr_tfidf.fit(self.X_train, self.y_train)

        print 'Best parameter set: %s ' % gs_lr_tfidf.best_params_
        print 'CV Accuracy: %.3f ' % gs_lr_tfidf.best_score_
        clf = gs_lr_tfidf.best_estimator_
        print 'Test Accuracy: %.3f' % clf.score(self.X_test, self.y_test)


def tokenizer_(text):
    return text.split()


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text)
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def tokenizer_porter(text):
    porter = PorterStemmer()
    snowball = SnowballStemmer(language='english')
    lancaster = LancasterStemmer()

    return [porter.stem(word) for word in text.split()]
    # return [snowball.stem(word) for word in text.split()]
    # return [lancaster.stem(word) for word in text.split()]
