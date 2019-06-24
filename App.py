#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask , render_template , request , redirect, url_for, jsonify
import numpy as np
from sklearn.externals import joblib
from nb_amazonReview import naive_bayes_sentiment
import os
current_path = os.getcwd()
app = Flask(__name__)



@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/predict',methods=["POST"])
def predict():
    text = request.form["input"]
    predictor = naive_bayes_sentiment()
    result = predictor.prediction(text)
    return render_template("prediction.html",result = result)


app.run(port = 6009 , debug=True)

