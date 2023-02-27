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
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)

    
    def is_weekend(tx_datetime):
        try:
            weekday=tx_datetime.weekday()
            is_weeekend = weekday>=5
            return int(is_weeekend)
        except Exception as e:
            raise FraudException(e, sys)


    def transferd_datetime(df):
        try:
            df['TX_DURING_WEEKEND']=df.TX_DATETIME.apply(is_weekend)
            return df 
        except Exception as e:
            raise FraudException(e, sys)


    
    def is_night(df,variable):
        try:
            for i in range(len(df[variable])):
                hour=df[variable][i].hour
                if hour<=6:
                    return 1
                else:
                    return 0
        except Exception as e:
            raise FraudException(e, sys)



    def transformed_day(df):
        try:
            df['TX_DURING_NIGHT']=is_night(df,'TX_DATETIME')
            return df 
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

    def customer_features(df):
        try:
            df=df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
            df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
            return df
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


    def get_transaction(df):
        try:
            df=df.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
            df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
            return df
        except Exception as e:
            raise FraudException(e, sys)
    
    def drop_index_column(data):
        try:
            if 'Unnamed: 0.1' in data.columns:
                data.drop('Unnamed: 0.1',axis=1,inplace=True)
            if  'Unnamed: 0' in data.columns:
                data.drop( 'Unnamed: 0',axis=1,inplace=True)
            return data
        except Exception as e:
            raise FraudException(e, sys)
    
    def reset_index(df):
        try:
            df=df.reset_index(inplace=False)
            return df
        except Exception as e:
            raise FraudException(e, sys) 

    @classmethod
    def get_data_transformer_object(cls):
        try:
            drop_duplicate=FunctionTransformer(drop_index_column,validate=False)
            get_weekend=FunctionTransformer(transferd_datetime,validate=False)
            get_day=FunctionTransformer(transformed_day,validate=False)
            get_customer=FunctionTransformer(customer_features,validate=False)
            get_transaction=FunctionTransformer(get_transaction,validate=False)
            get_reset_index=FunctionTransformer(reset_index,vcalidate=False)
            pipeline=Pipeline(steps[('drop_duplicates',drop_duplicate),
                                     ('weekday_or_weekend',get_weekend),
                                     ('day_or_night',get_day),
                                     ('customer_features',get_customer),
                                     ('transaction_features',get_transaction)
                                     ('reset_index',get_reset_index)])

        except Exception as e:
            raise FraudException(e, sys)





   
    def initiate_data_transformation(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)

            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            target_feature_train_df=train_df[TARGET_COLUMN]
            target_feature_test_df=test_df[TARGET_COLUMN]

            transformation_pipeline=DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)  

            input_feature_train_arr=transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr=transformation_pipeline.transform(input_feature_test_df)



            smt=SMOTETomek(sampling_strategy="minority")
            smt=SMOTETomek(random_state=77)

            input_feature_train_arr,target_feature_train_arr=smt.fit_resample(input_feature_train_arr,target_feature_train_df)

            input_feature_test_arr,target_feature_test_arr=smt.fit_resample(input_feature_test_arr,target_feature_test_df)


            train_arr=np.c_[input_feature_train_df,target_feature_train_arr]
            test_arr=np.c_[input_feature_test_df,target_feature_test_arr]


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

    
    
            

    
