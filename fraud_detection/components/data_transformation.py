from fraud_detection.entity import artifact_entity,config_entity 
from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException
from typing import Optional 
import pandas as pd 
import numpy as np 
from fraud_detection import utils 
import os,sys 
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from fraud_detection.config import TARGET_COLUMN
from sklearn.preprocessing import FunctionTransformer
from fraud_detection.preprocessing import is_week,is_day,customer_feature,terminal_feature,drop_columns


class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)


    @classmethod
    def get_data_transformation(cls):
        try:
            logging.info(f"Creating a pipline using Custom function with function transformer")
            get_week_end=FunctionTransformer(is_week, validate=False)
            get_is_day=FunctionTransformer(is_day, validate=False) 
            get_customer_features=FunctionTransformer(customer_feature, validate=False)
            get_terminal_features=FunctionTransformer(terminal_feature, validate=False) 
            get_columns=FunctionTransformer(drop_columns,validate=False)
            
            

  
            pipeline=Pipeline(steps=[
                ('week_days',get_week_end),
                ('day_night',get_is_day),
                ('cust_features',get_customer_features),
                ('term_features',get_terminal_features),
                ('drop_columns',get_columns)])

            return pipeline


        except Exception as e:
            raise FraudException(e, sys)


        
      
    def initiate_data_transformation(self):
        try:
            logging.info(f"reading training data set and converting column types into their respective data types")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df['TX_DATETIME'] = train_df['TX_DATETIME'].astype('datetime64[ns]')
            train_df['CUSTOMER_ID'] = train_df['CUSTOMER_ID'].astype('object')
            train_df['TERMINAL_ID'] = train_df['TERMINAL_ID'].astype('object')
            train_df['TX_TIME_SECONDS'] = train_df['TX_TIME_SECONDS'].astype('object')
            train_df['TX_TIME_DAYS'] = train_df['TX_TIME_DAYS'].astype('object')

            logging.info(f"reading training data set and converting column types into their respective data types")
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['TX_DATETIME'] = test_df['TX_DATETIME'].astype('datetime64[ns]')
            test_df['CUSTOMER_ID'] = test_df['CUSTOMER_ID'].astype('object')
            test_df['TERMINAL_ID'] = test_df['TERMINAL_ID'].astype('object')
            test_df['TX_TIME_SECONDS'] = test_df['TX_TIME_SECONDS'].astype('object')
            test_df['TX_TIME_DAYS'] = test_df['TX_TIME_DAYS'].astype('object')

            logging.info(f"Defining input for train and test datasets")
            input_train=train_df.drop(TARGET_COLUMN,axis=1)
            input_test=test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f"Defining target colum for both train test dataset")
            target_train=train_df[TARGET_COLUMN]
            target_test=test_df[TARGET_COLUMN]

            logging.info(f"Creating a transformation pipeline ")
            transformation_pipeline=DataTransformation.get_data_transformation()
            transformation_pipeline.fit(input_train)
            
            logging.info(f"fit and transform pipeline for both train and test datasets")
            input_train_arr=transformation_pipeline.transform(input_train)
            input_test_arr=transformation_pipeline.transform(input_test)

            smt=SMOTETomek(sampling_strategy="minority")
            smt=SMOTETomek(random_state=72)

            logging.info(f"This data set has imbalance data so applying Sampling technique for train and test data ")
            input_train_arr,target_train=smt.fit_resample(input_train_arr,target_train)
            input_test_arr,target_test=smt.fit_resample(input_test_arr,target_test)
            
            logging.info(f"Converting train and test data sets into numpy array to store")
            train_arr=np.c_[input_train_arr,target_train]
            test_arr=np.c_[input_test_arr,target_test]

            logging.info(f"storing converted train arr into transformed train path")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)

            logging.info(f"storing converted test arr into transformed train path")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            logging.info(f"storing transormation pipeline in transformer path , to use in future")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            logging.info(f'Preparing data transformation artifact')
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)
                

            logging.info(f"returning data transformation artifact")
            return data_transformation_artifact



        except Exception as e:
            raise FraudException(e, sys)

    
    
            

    
