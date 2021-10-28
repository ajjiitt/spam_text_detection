import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
model = joblib.load('./temp/model_sem5')
cv = joblib.load('./temp/cv_sem5')

def res(pred):
    if(pred == 0):
        print("NOT SPAM")
        return "NOT SPAM"
    else:
        print("SPAM")
        return "SPAM"

def ipTransformText(text):
    transformed_predict_text = cv.transform(text)
    arr = model.predict_proba(transformed_predict_text)
    print("NOT SPAM percentage", arr[0][0]*100, "%")
    print("SPAM percentage", arr[0][1]*100, "%")
    value = arr[0][1]*100
    return res(model.predict(transformed_predict_text))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('index.html', prediction_text=ipTransformText(request.form.values()))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    return jsonify({'prediction': ipTransformText([request.get_json(force=True)['text']])})

print("running @port 80")
app.run(debug=True, port=80)