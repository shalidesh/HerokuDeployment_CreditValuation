from flask import Flask,request,render_template,jsonify
import joblib
import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import os


app = Flask(__name__)

def loadModel():
    model = joblib.load(open("models/bajaj_3w_pipeline03.pkl",'rb'))
    return model

@app.route('/')
def index():

    return render_template('home.html')


    # return {"error": "Page not found"}

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():

    global model
    finalPred=0

    if request.method=='GET':
        return render_template('index.html', prediction=finalPred)
        
        # return {"error": "Not Allowed Method"}
           
    else:
        try:
            model=loadModel()
        except Exception as e:
            print("Load Model error",e)
            

        # # Get JSON data from the request body
        # data = request.json
        # print(data)

        # # Access specific data fields from the JSON object
        # age = data.get('age')
        # mileage = data.get('mileage')
        # strock = data.get('strock')

        # age = request.args.get('age')
        # mileage = request.args.get('mileage')
        # stroke = request.args.get('stroke')

        age = request.form.get("age")
        mileage = request.form.get("mileage")
        stroke = request.form.get("stroke")

        print(type(age))
        print(type(mileage))

        input_df = pd.DataFrame({'Age':[int(age)],'mileage':[int(mileage)],'stroke_values':[stroke]})
        prediction = model.predict(input_df)
        finalPred = int(np.round(prediction))

        return render_template('index.html', prediction=finalPred)


        # return {"Success": finalPred}


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)      

