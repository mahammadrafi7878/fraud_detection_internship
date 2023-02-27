from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from fraud_detection.config import mongo_db
import pandas as pd 
import numpy as np 
import os,sys 
import yaml 
import dill 


def get_collection_as_dataframe(database_name:str,collection_name):
    try:
        df=pd.DataFrame(list(mongo_db[database_name][collection_name].find()))
        if '_id' in df.columns:
            df.drop('_id',axis=1,inplace=True)
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        raise FraudException(e, sys)


def write_yaml_file(file_path,data):
    try:
        file_dir=os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)

        with open(file_path,'w') as file:
            yaml.dump(data,file)
        
    except Exception as e:
        raise FraudException(e, sys)  


def convert_columns_float(df,exclude_columns):
    try:
        for col in df.columns:
            if col not in exclude_columns:
                df[col]=df[col].astype(float)
        return df
    except Exception as e:
        raise FraudException(e, sys)   




def save_object(file_path:str,obj:object):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj) 
    except Exception as e:
        raise FraudException(e, sys) 


def load_object(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"the file {file_path} not exist")
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FraudException(e, sys) 




def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise FraudException(e, sys) 




def load_numpy_array_data(file_path:str):
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FraudException(e, sys)

