import os 
from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from fraud_detection.entity.config_entity import TRANSFER_OBJECT_FILE_NAME,MODEL_FILE_NAME
from glob import glob 
from typing import Optional 



class ModelResolver:
    def __init__(self,model_registery:str="saved_models",transformer_dir_name="transformer",model_dir_name='model'):
        try:
            self.model_registery=model_registery
            os.makedirs(self.model_registery,exist_ok=True)
            self.transformer_dir_name=transformer_dir_name
            self.model_dir_name=model_dir_name
        except Exception as e:
            raise FraudException(e,sys)   

    def get_latest_dir_path(self):
        try:
            dir_names=os.listdir(self.model_registery)
            if len(dir_names)==0:
                return None
            dir_names=list(map(int,dir_names))
            latest_dir_name=max(dir_names)
            return os.path.join(self.model_registery,f"{latest_dir_name}")
        except Exception as e:
            raise FraudException(e, sys)


    def get_latest_model_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME) 
        except Exception as e:
            raise FraudException(e, sys)


    def get_latest_transformer_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not avaialable")
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFER_OBJECT_FILE_NAME)
        except Exception as e:
            raise FraudException(e,sys)

    
    def get_latest_save_dir_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir ==None:
                return os.path.join(self.model_registery,f"{0}")
            latest_dir_num=int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registery,f"{latest_dir_num+1}") 
        except Exception as e:
            raise FraudException(e, sys) 

    def get_latest_save_model_path(self):
        try:
            latest_dir=self.get_latest_save_dir_path() 
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise FraudException(e, sys) 

    def get_latest_save_transformer_path(self):
        try:
            latest_dir=self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,TRANSFER_OBJECT_FILE_NAME) 
        except Exception as e:
            raise FraudException(e,sys)