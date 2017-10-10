#!/usr/bin/env python3
# coding: utf-8
# -*- coding: utf-8 -*-
import sys

from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_script import Manager
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    name, password, like = None, None, None
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
