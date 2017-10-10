from flask_wtf import Form
from wtforms import StringField, SubmitField, PasswordField,BooleanField
from wtforms import validators

class NameForm(Form):
    name = StringField(u'Name', render_kw={"placeholder":"What is your name?"}, validators=[validators.required()])
    password = PasswordField(u'password', render_kw={"placeholder": "password"}, validators=[validators.required()])
    like = BooleanField(u'like?')
    submit = SubmitField(u'Submit')


# **standard form fields**
# StringField
# TextAreaField
# PasswordField
# HiddenField
# DateField
# DateTimeField
# IntegerField
# DecimalField
# FloatField
# BooleanField
# RadioField
# SelectField
# SelectMultipleField
# FileField
# SubmitField
# FormFIeld
# FieldList
#
# **standard validators**
# Email
# EquailTo
# IPAddress
# Length
# NUmberRange
# Optional
# Required
# Reqexp
# URL
# Anyof
# Noneof




