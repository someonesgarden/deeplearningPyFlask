#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys

from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
import jinja2
import json


# define application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'atala0628'
moment = Moment(app)
manager = Manager(app)
bootstrap = Bootstrap(app)
site_title = "funwithdata:Deep"


@app.route('/', methods=['GET', 'POST'])
def index():
    name, password, like = None, None, None
    return render_template('index.html')


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


if __name__ == '__main__':
    app.debug = True
    app.run()
