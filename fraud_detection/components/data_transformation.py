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
from fraud_detection.preprocessing import transferd_datetime,transformed_day,customer_features,get_transaction,reset_index



class DataTransformation:
    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact

        except Exception as e:
            raise FraudException(e, sys)

    
    @classmethod
    def get_data_transformer_object(cls):
        try:
            get_weekend=FunctionTransformer(transferd_datetime,validate=False)
            get_day=FunctionTransformer(transformed_day,validate=False)
            get_customer=FunctionTransformer(customer_features,validate=False)
            get_transaction_data=FunctionTransformer(get_transaction,validate=False)
            get_reset_index=FunctionTransformer(reset_index,validate=False)
            pipeline=Pipeline(steps=[('weekday_or_weekend',get_weekend),
                                     ('day_or_night',get_day),
                                     ('customer_features',get_customer),
                                     ('transaction_features',get_transaction_data),
                                     ('reset_index',get_reset_index)])
            return pipeline

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
            transformation_pipeline.fit(target_feature_train_df)


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

    
    
            

    
