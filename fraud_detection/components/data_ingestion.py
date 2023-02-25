import os,sys 
from fraud_detection.logger import logging
from fraud_detection.exception import FraudException 
from fraud_detection.entity import config_entity
from fraud_detection.entity import artifact_entity 
from sklearn.model_selection import train_test_split 
import pandas as pd 
import numpy as np 
from fraud_detection import utils 


class DataIngestion:
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise FraudException(e, sys) 


    def initiate_data_ingestion(self):
        try:
            df=utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)

            df.replace(to_replace="na",value=np.NAN,inplace=True)

            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_path,index=False,header=True)

            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size)

            dataset_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)

            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)

            return data_ingestion_artifact
        except Exception as e:
            raise FraudException(e,sys)