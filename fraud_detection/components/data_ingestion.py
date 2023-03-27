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
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise FraudException(e, sys) 


    def initiate_data_ingestion(self):
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            df=utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"replacing na values with np.NAN")
            df.replace(to_replace="na",value=np.NAN,inplace=True)
            
            logging.info(f"Creating feature store path and directory if not available")
            feature_store_dir=os.path.dirname(self.data_ingestion_config.feature_store_path)
            os.makedirs(feature_store_dir,exist_ok=True)

            logging.info(f"storing data collections into feature store ")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_path,index=False,header=True)
            
            logging.info(f"splitting train and test parts")
            train_df,test_df=train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=777)

            logging.info(f'making target variable at last  ,beacuse we need to perform feature engineering so cant divide3 input and output features for both test and train data ')
            train_df['NEW_TX_FRAUD']=train_df['TX_FRAUD'].astype('int')
            test_df['NEW_TX_FRAUD']=test_df['TX_FRAUD'].astype('int')
            

            logging.info(f'making dataset dirctory to store train and test datasets')
            dataset_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)
            
            logging.info(f"storing train data set in dataset directory")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            
            logging.info(f"storing test dataset in dataset directory")
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            logging.info(f"Creating data ingestion artifact")
            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(
                feature_store_path=self.data_ingestion_config.feature_store_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            
            logging.info(f"returning data ingestion artifact")
            return data_ingestion_artifact
        except Exception as e:
            raise FraudException(e,sys)