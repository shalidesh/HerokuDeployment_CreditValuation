from flask import Flask,request,render_template,jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import datetime


import os


app = Flask(__name__)

def loadModel():
    model = joblib.load(open("models/bajaj_3w_pipeline03.pkl",'rb'))
    return model

@app.route('/')
def index():
    return render_template('bajaj.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    global model
    finalPred = 0

    curr_year = datetime.datetime.now().year
    print(curr_year)
    print(type(curr_year))

    try:
        model = loadModel()
    except Exception as e:
        print("Load Model error", e)

    if request.method == 'POST':
        if request.is_json:
            print("json request")  # check if the request data is of type json
            data = request.get_json(force=True)  # get data from JSON body
            yom = data.get('yom')
            mileage = data.get('mileage')
            stroke = data.get('stroke')

            age  = curr_year - int(yom) if yom else 0
        else:  # if not json, it's form data
            print("form request")
            yom = request.form.get("yom")
            mileage = request.form.get("mileage")
            mileage=int(mileage)
            stroke = request.form.get("stroke")
            
            age  = curr_year - int(yom) if yom else 0

        print(f'age is {type(age)}\nmileage is {type(mileage)}\n stroke is {type(stroke)}')
        
        print(f'inputs are :{age},{mileage},{stroke} and types are :{type(age)} ,{type(mileage)},{type(stroke)}')

        input_df = pd.DataFrame({'Age': [age], 'mileage': [int(mileage)], 'stroke_values': [stroke]})
        prediction = model.predict(input_df)
        finalPred = int(np.round(prediction))

        if request.is_json:  # if request was json, return json response
            return jsonify({'prediction': finalPred})
        else:  # else return normal template response
            return render_template('bajaj.html', prediction=finalPred)

    elif request.method == 'GET':
        return render_template('bajaj.html', prediction=finalPred)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)      

