#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys
path2proj = "/Users/user/PycharmProjects/DeepLearningPy"
sys.path.append(path2proj)

# ---------------------------- IMPORT::VENDOR ---------------------

from flask import Flask, request, render_template, url_for, jsonify
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
import jinja2
import json


# ---------------------------- IMPORT::CUSTOM ---------------------

# ---------------------------- IMPORT::DEEPLEARNING ---------------

import app.main.sentiment_analysis_app as saa

# ---------------------------- DEFINE APPLICATION ------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'atala0628'
app.debug = True
app.jinja_env.add_extension('pyjade.ext.jinja.PyJadeExtension')

moment = Moment(app)
manager = Manager(app)
bootstrap = Bootstrap(app)
site_title = "funwithdata:Deep"


# ---------------------------- FAVICON ------------------------------
@app.route('/favicon.ico')
def favicon():
    return (url_for('static', filename='favicon.ico'))

# ---------------------------- ROUTE --------------------------------


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('basic.jade', essentials={"site_title":"DeepLearningSomeonesgarden"}, sentiment="test")


@app.route('/aframe', methods=['GET', 'POST'])
def aframe():
    return render_template('aframe.jade', essentials={"site_title":"DeepLearningSomeonesgarden"}, sentiment="test")


@app.route('/main', methods=['GET', 'POST'])
def main():
    return render_template('main.jade', essentials={"site_title":"DeepLearningSomeonesgarden"}, sentiment="test")


@app.route('/math')
def math():
    essentials = {
        'utc_nowtime': datetime.utcnow(),
        'site_title': site_title,
        'title': "MATH.",
        'subtitle': ""
    }
    return render_template('math.jade', essentials=essentials, test1=123)


@app.route('/analys1', methods=['GET','POST'])
def analys1():
    if request.method == 'POST':
        form = request.form
    else:
        form = request.args
    param = json.loads(form.get('param'))


    return jsonify({'action':'juc'})


@app.route('/sentimentanalysis', methods=['GET', 'POST'])
def sentimentanalysis():
    if request.method == 'POST':
        form = request.form
    else:
        form = request.args
    param = json.loads(form.get('param'))
    res = {}

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
    return render_template('404.jade', error=err), 404


@app.errorhandler(ValueError)
@app.errorhandler(UnicodeDecodeError)
@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def error_handler(err):
    return render_template('404.jade', error=err)

# ---------------------------- MAIN ----------------------------------


if __name__ == '__main__':
    app.run()
