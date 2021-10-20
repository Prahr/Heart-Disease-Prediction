from flask import Flask, request, render_template
import numpy as np
from joblib import load
app = Flask(__name__)

@app.route('/')
def index():
     return render_template('index.html')
def predict(input_list):
     input = np.array(input_list).reshape(1, 11)
     model = load("model.joblib")
     preds = model.predict_proba(input)
     return preds[0, 0]
@app.route('/results', methods = ['POST'])
def results():
     if request.method == 'POST':
          input_list = request.form.to_dict()
          input_list = list((input_list.values()))
          input_list = list(map(float, input_list))
          result = predict(input_list)
          prediction = str(round(result*100, 2)) + "%"
          return render_template("results.html", prediction=prediction)
@app.route('/data')
def data():
     return render_template('data.html')