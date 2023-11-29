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
from pipelines.predict_pipeline import CustomData,CustomDataAutoFinance,PredictPipeline
import os
import json


app = Flask(__name__)


df=pd.read_csv('datasets/Suzuki_with_sub_models_new_v6.csv')
df=df[['Age','Mileage','Engine_Capacity','Model','Fuel_Type','Transmission','Price']]

model_options = df['Model'].unique()
fuel_type_options = df['Fuel_Type'].unique()
transmission_options = df['Transmission'].unique()


df1=pd.read_csv('datasets/foretest_Toyota_v2.csv')
df1=df1[['Age','Mileage','Engine_Capacity','Model','Fuel_Type','Transmission','Price']]

model_options1 = df1['Model'].unique()
fuel_type_options1 = df1['Fuel_Type'].unique()
transmission_options1 = df1['Transmission'].unique()

model_path_location_3w=os.path.join("models","bajaj")
model_path_location_suzuki=os.path.join("models","suzuki")
model_path_location_toyota=os.path.join("models","toyota")



@app.route('/')
def index():
    return render_template('bajaj.html')


@app.route('/suzuki')
def suzuki():
    return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options)

@app.route('/toyota')
def toyota():
    return render_template('suzuki.html',model_options=model_options1,fuel_type_options=fuel_type_options1,transmission_options=transmission_options1)



@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    manufacture = request.form.get('manufacture')
    
    if request.method == 'POST':

        if manufacture=="bajaj":

            if request.is_json:
                return render_template('bajaj.html')
                
            else: 
                data=CustomData(
                    yom=request.form.get('yom'),
                    milage=request.form.get('mileage'),
                    strock=request.form.get('stroke'),
                    light_type=request.form.get('light'),

                )

                pred_df=data.get_data_as_data_frame()
        
                predict_pipeline=PredictPipeline(model_path_location=model_path_location_3w)
                
                results=predict_pipeline.predict(pred_df)
                

                return render_template('bajaj.html', prediction=int(np.round(results[0])))

        elif manufacture=="suzuki":

            if request.is_json:
                 return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options)  
            else: 
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
              
                predict_pipeline=PredictPipeline(model_path_location=model_path_location_suzuki)
                
                results=predict_pipeline.predict(pred_df)
                
                return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options, prediction=int(np.round(results[0])))

        elif manufacture=="toyota":

            if request.is_json:
                return render_template('toyota.html', model_options=model_options1,fuel_type_options=fuel_type_options1,transmission_options=transmission_options1)
            else: 
                yom=request.form.get('yom')
                
                with open('constant/constant_toyoto.json') as json_file:
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
                
                predict_pipeline=PredictPipeline(model_path_location=model_path_location_toyota)
                
                results=predict_pipeline.predict(pred_df)
                
                

                return render_template('toyota.html', prediction=int(np.round(results[0])))


    elif request.method == 'GET':
        return render_template('bajaj.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)      

