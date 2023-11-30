from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import datetime
from predict_pipeline import CustomDataAutoFinance,PredictPipeline
import os
import json


app = Flask(__name__)


df=pd.read_csv('datasets/Suzuki_with_sub_models_new_v6.csv')
df=df[['Age','Mileage','Engine_Capacity','Model','Fuel_Type','Transmission','Price']]

model_options = df['Model'].unique()
fuel_type_options = df['Fuel_Type'].unique()
transmission_options = df['Transmission'].unique()


@app.route('/')
def index():
    return render_template('bajaj.html')


@app.route('/suzuki')
def suzuki():
    return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    
    if request.method == 'POST':
                   
        yom=request.form.get('yom')

        with open('constant/constants_suzuki.json') as json_file:
            data = json.load(json_file)

        milage=data[yom]
        print(milage)
        
        data=CustomDataAutoFinance(
            yom=request.form.get('yom'),
            model=request.form.get('model'),
            milage=milage,
            engine_capacity=request.form.get('engine_capacity'),
            fuel=request.form.get('fuel'),
            transmission=request.form.get('transmission')

        )
    
        pred_df=data.get_data_as_data_frame()
        
        predict_pipeline=PredictPipeline()
        
        results=predict_pipeline.predict(pred_df)
        
        return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options, prediction=int(np.round(results[0])))


        

    elif request.method == 'GET':
        return render_template('suzuki.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port,debug=True)    

