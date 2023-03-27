from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
import os,sys 
import pandas as pd 
import numpy as np 
from datetime import datetime 
from fraud_detection.utils import load_object 
from fraud_detection.predictor import ModelResolver 

PREDICTION_DIR='predictor'

def satrt_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace=True)
        df['TX_DATETIME'] = df['TX_DATETIME'].astype('datetime64[ns]')
        df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype('object')
        df['TERMINAL_ID'] = df['TERMINAL_ID'].astype('object')
        df['TX_TIME_SECONDS'] = df['TX_TIME_SECONDS'].astype('object')
        df['TX_TIME_DAYS'] = df['TX_TIME_DAYS'].astype('object')
        #validation
        
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        
        
        df["prediction"]=prediction
       


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
        
    except Exception as e:
        raise FraudException(e, sys)




            

        