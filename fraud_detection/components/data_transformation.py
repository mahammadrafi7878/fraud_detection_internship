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
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline



class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifactdata_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)

    
    def is_weekend(df,variable):
        for i in range(len(df[variable])):
            weekday=df[variable][i].weekday()
            if weekday >5:
                return 1
            else:
                return 0


    
    def is_night(df,variable):
        for i in range(len(df[variable])):
            hour=df[variable][i].hour
            if hour<=6:
                return 1
            else:
                return 0  

    
    def get_customer_spending_behaviour_features(customer_transactions,window_sizes_in_days=[1,7,30]):
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


    def get_count_risk_rolling_window(terminal_transactions,delay_period,windows_sizes_in_days=[1,7,30]):
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
    
    def drop_index_column(data):
        if 'Unnamed: 0.1' in data.columns:
            data.drop('Unnamed: 0.1',axis=1,inplace=True)
        if  'Unnamed: 0' in data.columns:
            data.drop( 'Unnamed: 0',axis=1,inplace=True)
        return data



    def initiate_data_transformation(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)


            train_df['TX_DURING_WEEKEND']=is_weekend(tarin_df,'TX_DATETIME')
            test_df['TX_DURING_WEEKEND']=is_weekend(test_df,'TX_DATETIME') 

            train_df['TX_DURING_NIGHT']=is_night(train_df,'TX_DATETIME')
            test_df['TX_DURING_NIGHT']=is_night(test_df,'TX_DATETIME')

            train_df=train_df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
            train_df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
            test_df=test_df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
            test_df=test_df.sort_values('TX_DATETIME').reset_index(drop=True)


            tarin_df=train_df.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            train_df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
            test_df=test_df.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            test_df=test_df.sort_values('TX_DATETIME').reset_index(drop=True)


            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            target_feature_train_df=tarin_df[TARGET_COLUMN]
            target_feature_test_df=test_df[TARGET_COLUMN]


            smt=SMOTETomek(sampling_strategy="minority")
            smt=SMOTETomek(random_state=77)

            input_feature_train_df,target_feature_train_df=smt.fit_resample(input_feature_train_df,target_feature_train_df)

            input_feature_test_df,target_feature_test_df-smt.fit_resample(input_feature_test_df,target_feature_test_df)


            train_arr=np.c_[input_feature_train_df,target_feature_train_df]
            test_arr=np.c_[input_feature_test_df,target_feature_test_df]



        except Exception as e:
            raise FraudException(e, sys)

    
    
            

    
