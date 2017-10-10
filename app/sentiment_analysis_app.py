#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys
import os
import importlib
import sqlite3
path2proj = "/Users/user/PycharmProjects/deeplearningFlask"
sys.path.append(path2proj)
importlib.reload(sys)
import pickle
import numpy as np
from app.movieclassifier.vectorizer import vect


def create_sql_examples():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    try:
        c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')
    except:
        print("error")

    example1 = 'I love this movie'
    c.execute("INSERT INTO review_db"
              " (review, sentiment,date) VALUES (?, ?, DATETIME('now'))", (example1, 1))
    example2 = 'I disliked this movie'
    c.execute("INSERT INTO review_db"
              " (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (example2, 0))
    conn.commit()

    # CHECK DATA
    c.execute("SELECT * FROM review_db WHERE date BETWEEN '2016-01-01 00:00:00' AND DATETIME('now')")
    results = c.fetchall()
    conn.commit()
    conn.close()

    print(results)


def show_review_db():
    conn = sqlite3.connect(db)
    c = conn.cursor()
    try:
        c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')
    except:
        print("table aready exists. no action.")
    # CHECK DATA
    c.execute("SELECT * FROM review_db WHERE date BETWEEN '2016-01-01 00:00:00' AND DATETIME('now')")
    results = c.fetchall()
    conn.commit()
    conn.close()

    print(results)


def classify(document):
    label = {
        0: 'negative',
        1: 'positive'
    }
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])
    print("train")
    print(X)
    print([y])


def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    print(c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y)))
    conn.commit()
    conn.close()

def update_model(path, model, batch_size=10000):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT * from review_db")

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)
        print("update_model")
        print(X)
        print(y)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return None

# ----------------

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'movieclassifier/pkl_objects/classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
update_model(path=db, model=clf, batch_size=10000)