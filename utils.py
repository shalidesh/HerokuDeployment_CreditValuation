import numpy as np 
import pandas as pd
import pickle
import json
import os
from flask import request




def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        print("error occured at model loading",e)

def getVehicleModels(df_name:str):

    try:
        df=pd.read_csv(f'datasets/{df_name}')
        df=df[['Age','Mileage','Engine_Capacity','Model','Fuel_Type','Transmission','Price']]

        model_options = df['Model'].unique()
        fuel_type_options = df['Fuel_Type'].unique()
        transmission_options = df['Transmission'].unique()

    except Exception as e:
        print("vehicle models selection error in suzuki",e)

    return model_options,fuel_type_options,transmission_options


def getMileage(jasonpath:str):
    with open(f'constant/{jasonpath}') as json_file:
            data = json.load(json_file)

    return data



