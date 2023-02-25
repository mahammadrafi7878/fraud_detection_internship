import os,sys 
from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException 
from datetime import datetime

FILE_NAME="fraud.csv"
TRAIN_FILE_NAME='train.csv'
TEST_FILE_NAME='test.csv'

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir=os.path.join(os.getcwd(),'artifacts',f"{datetime.now().strftime('%d%m%Y__%H%M%S')}")
        except Exception as e:
            raise FraudException(e, sys)




class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name='fraud_detection'
            self.collection_name='predicting_fraud'
            self.data_ingestion_dir=os.path.join(training_pipeline_config.artifact_dir,'data_ingestion')
            self.feature_store_path=os.path.join(self.data_ingestion_dir,'feature_store',FILE_NAME)
            self.train_file_path=os.path.join(self.data_ingestion_dir,'dataset',TRAIN_FILE_NAME)
            self.test_file_path=os.path.join(self.data_ingestion_dir,'dataset',TEST_FILE_NAME)
            self.test_size=0.30

        except Exception as e:
            raise FraudException(e, sys)

    def to_dict(self):
        try:
            return self.__dict__
        except Exception as e:
            raise FraudException(e, sys)



class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir=os.path.join(training_pipeline_config.artifact_dir,'data_valiadtion')
        self.report_file_path=os.path.join(self.data_validation_dir,'report.yaml')
        self.base_file_path=os.path.join('total_data.csv')
        self.missing_threshold=0.2
class DataTransformationConfig:...
class ModelTrainerConfig:...
class ModelEvaluationConfig:...
class ModelPusherConfig:...