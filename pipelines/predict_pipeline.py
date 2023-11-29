import sys
import pandas as pd
from exception import CustomException
from utils import load_object
import os
import datetime


class PredictPipeline:
    def __init__(self,model_path_location:str):
        self.model_path_location=model_path_location

    def predict(self,features):
        try:
            model_path=os.path.join(self.model_path_location,"model.pkl")
            preprocessor_path=os.path.join(self.model_path_location,"preprocessor.pkl")
            print("model before loading")
            model=load_object(file_path=model_path)
            print("model after loading")
            print("preproceesor before loading")
            preprocessor=load_object(file_path=preprocessor_path)
            print("preproceesor after loading")
            data_scaled=preprocessor.transform(features)
            # preds=model.predict(data_scaled)
            return 500000
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        yom: str,
        milage: str,
        strock:str,
        light_type: str,
        ):

        self.mileage = milage

        self.yom = yom

        self.curr_year=datetime.datetime.now().year

        self.age=self.curr_year - int(self.yom) if int(self.yom) else 0

        self.strock = strock

        self.light_type = light_type


    def get_data_as_data_frame(self):
        try:

            custom_data_input_dict = {
                "mileage": [int(self.mileage)],
                "Age": [self.age],
                "stroke_values": [self.strock],
                "Light Type": [self.light_type]      
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class CustomDataAutoFinance:
    def __init__(self,
        yom: str,
        model:str,
        # sub_model:str,
        milage: str,
        engine_capacity:str,
        fuel:str,
        transmission: str
        ):

        self.mileage = milage

        self.yom = yom

        self.curr_year=datetime.datetime.now().year

        self.age=self.curr_year - int(self.yom) if int(self.yom) else 0

        self.model = model

        self.engine_capacity = engine_capacity

        self.fuel=fuel

        self.transmission=transmission


    def get_data_as_data_frame(self):
        try:

            custom_data_input_dict = {
                # "Sub_Model":[self.sub_model],
                "Age": [self.age],
                "Mileage": [int(self.mileage)],
                "Engine_Capacity": [self.engine_capacity],
                "Model": [self.model],
                "Fuel_Type": [self.fuel],
                "Transmission": [self.transmission]        
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

