from fraud_detection.entity import artifact_entity,config_entity 
from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException
from typing import Optional 
import pandas as pd 
import numpy as np 
from fraud_detection import utils 
import os,sys 
from imblearn.combine import SMOTETomek
from fraud_detection.preprocessing import is_day,is_week,get_customer_spending_behaviour_features,get_count_risk_rolling_window


class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)

    def get_data_transformation(self,input_train,input_test):
        try:
            input_train=is_week(input_train)
            input_test=is_week(input_test)

            input_train=is_day(input_train)
            input_test=is_day(input_test)

            input_train=input_train.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,windows_sizes_in_days=[1,7,30]))
            input_train=input_train.sort_values('TX_DATETIME').reset_index(drop=True)
            input_test=input_test.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,windows_sizes_in_days=[1,7,30]))
            input_test=input_test.sort_values('TX_DATETIME').reset_index(drop=True)


            input_train=input_train.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            input_train=input_train.sort_values('TX_DATETIME').reset_index(drop=True)
            input_test=input_test.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            input_test=input_test.sort_values('TX_DATETIME').reset_index(drop=True) 


            return input_train,input_test

        except Exception as e:
            raise FraudException(e, sys)


        
      
    def initiate_data_transformation(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df['TX_DATETIME'] = train_df['TX_DATETIME'].astype('datetime64[ns]')
            train_df['CUSTOMER_ID'] = train_df['CUSTOMER_ID'].astype('object')
            train_df['TERMINAL_ID'] = train_df['TERMINAL_ID'].astype('object')
            train_df['TX_TIME_SECONDS'] = train_df['TX_TIME_SECONDS'].astype('object')
            train_df['TX_TIME_DAYS'] = train_df['TX_TIME_DAYS'].astype('object')


            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['TX_DATETIME'] = test_df['TX_DATETIME'].astype('datetime64[ns]')
            test_df['CUSTOMER_ID'] = test_df['CUSTOMER_ID'].astype('object')
            test_df['TERMINAL_ID'] = test_df['TERMINAL_ID'].astype('object')
            test_df['TX_TIME_SECONDS'] = test_df['TX_TIME_SECONDS'].astype('object')
            test_df['TX_TIME_DAYS'] = test_df['TX_TIME_DAYS'].astype('object')
            
            
            


            target_train=train_df["TX_FRAUD"]
            target_test=test_df["TX_FRAUD"]
            input_train=train_df
            input_test=test_df

            transformation_pipeline=self.get_data_transformation(input_train=input_train, input_test=input_test)
            input_train_arr,input_test_arr=transformation_pipeline



            input_train_arr=input_train_arr.drop('TX_DATETIME',axis=1)
            input_test_arr=input_test_arr.drop('TX_DATETIME',axis=1)
            print(input_train_arr.dtypes)

            input_train_arr['NEW_TX_FRAUD']=input_train_arr['TX_FRAUD'].astype('int')
            input_test_arr['NEW_TX_FRAUD']=input_test_arr['TX_FRAUD'].astype('int')

            input_train_arr.drop(['TX_FRAUD','TRANSACTION_ID','CUSTOMER_ID','TERMINAL_ID','TX_TIME_SECONDS','TX_TIME_DAYS','TX_FRAUD_SCENARIO'],axis=1,inplace=True)
            input_test_arr.drop(['TX_FRAUD','TRANSACTION_ID','CUSTOMER_ID','TERMINAL_ID','TX_TIME_SECONDS','TX_TIME_DAYS','TX_FRAUD_SCENARIO'],axis=1,inplace=True)
            print(input_train_arr.dtypes)

            



            smt=SMOTETomek(sampling_strategy="minority")
            smt=SMOTETomek(random_state=77)

            input_train_arr,target_train=smt.fit_resample(input_train_arr,target_train)

            input_test_arr,target_test=smt.fit_resample(input_test_arr,target_test)


            train_arr=np.c_[input_train_arr,target_train]
            test_arr=np.c_[input_test_arr,target_test]


            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)
                

            
            return data_transformation_artifact



        except Exception as e:
            raise FraudException(e, sys)

    
    
            

    
