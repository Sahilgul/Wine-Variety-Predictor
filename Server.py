from flask import Flask,render_template, request,jsonify

import numpy as np
import pandas as pd

from gensim import utils
import gensim.parsing.preprocessing as gsp

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

df2 = pd.read_csv('df2.csv')

X = df2.description
y = df2.variety
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=245)

#Creating Model
model = make_pipeline(CountVectorizer(max_features=11000),MultinomialNB())
#training the model
model.fit(X_train,y_train)

def predict_var(str):
    pred = model.predict([str])[0]
    return pred

app = Flask(__name__)

@app.route("//")
def home():
    # wine_var = df2['variety'].values
    # return render_template('web.html', locations=wine_var)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    descrip = request.form.get('descrip')
    pred = model.predict([descrip])[0]
    return pred

if __name__ == '__main__':
    app.run(debug=True)
