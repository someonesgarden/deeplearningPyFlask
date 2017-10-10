#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys
path2proj = "/Users/user/PycharmProjects/DeepLearningPy"
sys.path.append(path2proj)

# ---------------------------- IMPORT::VENDOR ---------------------

from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
import numpy as np
import jinja2
import json

import app.sentiment_analysis_app as saa


# ---------------------------- IMPORT::CUSTOM ---------------------

from lib.flask.Form1 import *

# ---------------------------- IMPORT::DEEPLEARNING ---------------

import model.DataPreprocessing as mdp
import model.ModelFitting as mmf
import app.sentiment_analysis_app as saa

# ---------------------------- DEFINE APPLICATION ------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'atala0628'
app.debug = True
app.jinja_env.add_extension('pyjade.ext.jinja.PyJadeExtension')

moment = Moment(app)
manager = Manager(app)
bootstrap = Bootstrap(app)
site_title = "funwithdata:Deep"

# ---------------------------- ROUTE --------------------------------


@app.route('/', methods=['GET', 'POST'])
def index():
    name, password, like = None, None, None
    return render_template('main.jade', essentials={"site_title":"サイトタイトル"}, sentiment="test")

@app.route('/sentimentanalysis', methods=['GET', 'POST'])
def sentimentanalysis():
    if request.method == 'POST':
        form = request.form
    else:
        form = request.args
    param = json.loads(form.get('param'))
    res = {}
    print(param)

    if param['action'] == 'analyse':
        review = param['review']
        label, proba = "", 0.0
        if review != "":
            label, proba = saa.classify(review)

        res['action'] = 'analyse'
        res['label'] = label
        res['proba'] = proba


    elif param['action'] == 'feedback':
        review = param['review']
        prediction = param['prediction']


    res['action'] = 'feedback'

    return jsonify(res)

# ---------------------------- ROUTE::ERROR -------------------------

@app.route('/404')
@app.errorhandler(404)
@app.errorhandler(500)
@app.errorhandler(503)
@app.errorhandler(504)
@app.errorhandler(501)
def error_handler(err):
    return render_template('404.html', error=err), 404

@app.errorhandler(ValueError)
@app.errorhandler(UnicodeDecodeError)
@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def error_handler(err):
    return render_template('404.html', error=err)

# ---------------------------- MAIN ----------------------------------


if __name__ == '__main__':
    app.run()
