from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from fraud_detection.config import mongo_db
import pandas as pd 
import numpy as np 
import os,sys 


def get_collection_as_dataframe(database_name:str,collection_name):
    try:
        df=pd.DataFrame(list(mongo_db[database_name][collection_name].find()))
        return df
    except Exception as e:
        raise FraudException(e, sys)