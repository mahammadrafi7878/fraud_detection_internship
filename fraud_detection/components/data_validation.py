from typing import Optional 
import os,sys 
import pandas as pd 
import numpy as np 
from fraud_detection import utils 
from fraud_detection.entity import artifact_entity,config_entity
from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from scipy.stats import ks_2samp 
from scipy.stats import chi2_contingency



class DataValidation:
    def __init__ (self, data_validation_config:config_entity.DataValidationConfig,
                        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict() 
        except Exception as e:
            raise FraudException(e, sys)

    



    def drop_missing_values_columns(self,df,report_key_name:str):
        try:
            threshold=self.data_validation_config.missing_threshold
            null_report=df.isna().sum()/df.shape[0]

            drop_column_names=null_report[null_report>threshold].index

            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            if len(df.columns)==0:
                return None 
            return df 
        except Exception as e:
            raise FraudException(e, sys)
    


    def is_required_columns_exist(self,base_df,report_key_name:str,current_df):
        try:
            base_columns=base_df.columns 
            current_columns=current_df.columns
            missing_columns=[]
            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

                if len(missing_columns)>0:
                    self.validation_error[report_key_name]=missing_columns
                    return False 

                return True
        except Exception as e:
            raise FraudException(e, sys)
    



    def data_drift_numerical(self,base_df,current_df,report_key_name):
        try:
            drift_report=dict()
            base_columns=base_df.columns
            current_columns=current_df.columns 

            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]

                same_distribution=ks_2samp(base_data, current_data) 


                if same_distribution.pvalue>0.05:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True

                    }
                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }

            self.validation_error[report_key_name]=drift_report


        except Exception as e:
            raise FraudException(e, sys)

    

    def data_drift_categorical(self,base_df,current_df,report_key_name):
        try:
            drift_report=dict()
            base_columns=base_df.columns
            current_columns=current_df.columns 

            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]

                same_distribution=chi2_contingency(base_data, current_data) 


                if same_distribution.pvalue>0.05:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True

                    }
                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }

            self.validation_error[report_key_name]=drift_report


        except Exception as e:
            raise FraudException(e, sys)
    
    



    def initiate_data_validation(self):
        try:
            base_df=pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN},inplace=True) 
            base_df['TX_DATETIME'].astype('object')
            
            base_df=self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_with in base_datase")


            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            
            train_df['TX_DATETIME'].astype('object')
            
            test_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df['TX_DATETIME'].astype('object')

            train_df=self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_in_train_df")
            test_df=self.drop_missing_values_columns(df=test_df, report_key_name="missing values in test df")
            

            
            exclude_columns=["CUSTOMER_ID","TERMINAL_ID","TX_TIME_SECONDS","TX_TIME_DAYS","TX_DATETIME",'TX_FRAUD_SCENARIO']  

            base_df=utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df=utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df=utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)



            train_df_column_status=self.is_required_columns_exist(base_df=base_df, report_key_name="missing_values_in_train_df", current_df=train_df)
            test_df_column_status=self.is_required_columns_exist(base_df=base_df, report_key_name="missing values in test df", current_df=test_df)
            

            base_df_numerical_columns=base_df[["TX_FRAUD","TX_AMOUNT","TRANSACTION_ID"]]
            train_df_numerical_columns=train_df[["TX_FRAUD","TX_AMOUNT","TRANSACTION_ID"]]
            test_df_numerical_columns=test_df[["TX_FRAUD","TX_AMOUNT","TRANSACTION_ID"]]

            base_df_categorical_columns=base_df[["CUSTOMER_ID","TERMINAL_ID","TX_TIME_SECONDS","TX_TIME_DAYS","TX_DATETIME"]] 
            train_df_categorical_columns=train_df[["CUSTOMER_ID","TERMINAL_ID","TX_TIME_SECONDS","TX_TIME_DAYS","TX_DATETIME"]] 
            test_df_categorical_columns=test_df[["CUSTOMER_ID","TERMINAL_ID","TX_TIME_SECONDS","TX_TIME_DAYS","TX_DATETIME"]] 



             
            if train_df_column_status:
                self.data_drift_numerical(base_df=base_df_numerical_columns, current_df=train_df_numerical_columns, report_key_name="data_drift_within_train_numerical_dataset")
                self.data_drift_categorical(base_df=base_df_categorical_columns, current_df=train_df_categorical_columns, report_key_name="data_drift_within_train_categorical_dataset")


            if test_df_column_status:
                self.data_drift_numerical(base_df=base_df_numerical_columns, current_df=test_df_numerical_columns, report_key_name="data_drift_within_test_numerical_dataset")
                self.data_drift_categorical(base_df=base_df_categorical_columns,current_df=test_df_categorical_columns,report_key_name="data_drift_within_test_categorical_dataset")

            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)



            data_validation_artifact=artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)

            return data_validation_artifact          

        except Exception as e:
            raise FraudException(e, sys)



    