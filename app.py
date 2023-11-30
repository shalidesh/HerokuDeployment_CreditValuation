from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pipelines.predict_pipeline import CustomDataAutoFinance,PredictPipeline
from utils import getVehicleModels,getMileage
import os
import json


app = Flask(__name__)


suzuki_dataset='Suzuki_with_sub_models_new_v6.csv'
toyota_dataset='foretest_Toyota_v2.csv'

suzuki_model_path=os.path.join('models',"suzuki")
toyota_model_path=os.path.join('models',"toyota")

suzuki_json='constants_suzuki.json'
toyota_json='constant_toyoto.json'


model_options,fuel_type_options,transmission_options=getVehicleModels(suzuki_dataset)
model_options_t,fuel_type_options_t,transmission_options_t=getVehicleModels(toyota_dataset)

@app.route('/')
def index():
    return render_template('bajaj.html')


@app.route('/suzuki')
def suzuki():
    
    return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options)


@app.route('/toyota')
def toyota():
    
     return render_template('toyota.html',model_options=model_options_t,fuel_type_options=fuel_type_options_t,transmission_options=transmission_options_t)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    manufacture = request.form.get('manufacture')

    if request.method == 'POST':

        if manufacture=="suzuki":

            yom=request.form.get('yom')

            jasonData=getMileage(suzuki_json)

            milage=jasonData[yom]
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
            
            predict_pipeline=PredictPipeline(suzuki_model_path)
            
            results=predict_pipeline.predict(pred_df)
                    
            return render_template('suzuki.html',model_options=model_options,fuel_type_options=fuel_type_options,transmission_options=transmission_options, prediction=int(np.round(results[0])))
    

        elif manufacture=="toyota":

            yom=request.form.get('yom')

            jasonData=getMileage(toyota_json)

            milage=jasonData[yom]
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
            
            predict_pipeline=PredictPipeline(toyota_model_path)
            
            results=predict_pipeline.predict(pred_df)
                    
            return render_template('toyota.html',model_options=model_options_t,fuel_type_options=fuel_type_options_t,transmission_options=transmission_options_t, prediction=int(np.round(results[0])))
        
        
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port,debug=True)    

