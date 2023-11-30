import os
import sys
import numpy as np 
import pandas as pd
import pickle



    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        print("error occured at model loading",e)