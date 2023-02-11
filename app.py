from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('auto.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    wheelbase = float(data['wheelbase'])
    curbweight = float(data['curbweight'])
    enginesize = float(data['enginesize'])
    citympg = float(data['citympg'])
    highwaympg = float(data['highwaympg'])


    test_array = np.array([[wheelbase,curbweight,enginesize,citympg,highwaympg]])

    pred = model.predict(test_array)
    print('hello')
    return render_template('after.html', Charges=np.round(pred[0],2))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)    