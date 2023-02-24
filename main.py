from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException
from fraud_detection.utils import get_collection_as_dataframe 
import pandas as pd 
import numpy as np 
import os,sys  


print(__name__)
if __name__ =='__main__':
    try:
        df=get_collection_as_dataframe(database_name='fraud_detection', collection_name='predicting_fraud')
        print(df.shape)
    except Exception as e:
        raise FraudException(e,sys)
    


