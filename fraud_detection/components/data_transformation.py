from fraud_detection.entity import artifact_entity,config_entity 
from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException
from typing import Optional 
import pandas as pd 
import numpy as np 
from fraud_detection import utils 
import os,sys 
from fraud_detection.config import TARGET_COLUMN
from imblearn.combine import SMOTETomek
from fraud_detection.preprocessing import is_day,is_week,get_customer_spending_behaviour_features,get_count_risk_rolling_window
class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)

    def is_weekend(df,variable):
        try:
            for i in range(len(df[variable])):
                week=df[variable][i].weekday
            if week >5:
                return 1
            else:
                 return 0
        except Exception as e:
            raise FraudException(e, sys)


        
    def is_night(df:pd.DataFrame,variable:str):
        try:
            for i in range(len(df[variable])):
                hour=df[variable][i].hour
                if hour<=6:
                    return 1
                else:
                    return 0
        except Exception as e:
            raise FraudException(e, sys)



        
    def get_customer_spending_behaviour_features(customer_transactions,window_sizes_in_days=[1,7,30]):
        try:
            customer_transactions=customer_transactions.sort_values('TX_DATETIME')
            customer_transactions.index=customer_transactions.TX_DATETIME
            for window_size in window_sizes_in_days:
                SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
                NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()
        
                AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW
        
                customer_transactions['COUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
                customer_transactions['COUSTOMER_ID_AVG_AMOUNT_TX_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
            customer_transactions.index=customer_transactions.TRANSACTION_ID
    
            return customer_transactions
        except Exception as e:
            raise FraudException(e, sys)

    
    def get_count_risk_rolling_window(terminal_transactions,delay_period,windows_sizes_in_days=[1,7,30]):
        try:
            terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')
            terminal_transactions.index=terminal_transactions.TX_DATETIME
    
            NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
            NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()
    
            for window_size in windows_sizes_in_days:
                NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
                NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()
        
                NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
                NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY 
        
        
                RISK_WINDOW=NB_FRAUD_WINDOW/NB_TX_WINDOW 
        
                terminal_transactions['TERMINAL_ID'+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
                terminal_transactions['TERMINAL_ID'+'_RISK'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)
        
            terminal_transactions.index=terminal_transactions.TRANSACTION_ID
            terminal_transactions.fillna(0,inplace=True)
            return terminal_transactions
        except Exception as e:
            raise FraudException(e, sys) 


        
      
    def initiate_data_transformation(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            train_df['TX_DATETIME'] = train_df['TX_DATETIME'].astype('datetime64[ns]')
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['TX_DATETIME'] = test_df['TX_DATETIME'].astype('datetime64[ns]')

            input_train=train_df.drop(TARGET_COLUMN,axis=1)

            input_test=test_df.drop(TARGET_COLUMN,axis=1)

            target_train=train_df[TARGET_COLUMN]
            target_test=test_df[TARGET_COLUMN]

            input_train=is_week(input_train)
            input_test=is_week(input_test)

            input_train=is_day(input_train)
            input_test=is_day(input_test)

            input_train=input_train.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
            input_train=input_train.sort_values('TX_DATETIME').reset_index(drop=True)
            input_test=input_test.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
            input_test=input_test.sort_values('TX_DATETIME').reset_index(drop=True)


            input_train=input_train.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            input_train=input_train.sort_values('TX_DATETIME').reset_index(drop=True)
            input_test=input_test.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            input_test=input_test.sort_values('TX_DATETIME').reset_index(drop=True) 




            smt=SMOTETomek(sampling_strategy="minority")
            smt=SMOTETomek(random_state=77)

            input_train,target_train=smt.fit_resample(input_train,target_train)

            input_test,target_test=smt.fit_resample(input_test,target_test)


            train_arr=np.c_[input_train,target_train]
            test_arr=np.c_[input_test,target_test]


            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)


            #utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path)
                

            
            return data_transformation_artifact



        except Exception as e:
            raise FraudException(e, sys)

    
    
            

    
