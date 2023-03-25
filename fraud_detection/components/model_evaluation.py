from fraud_detection.predictor import ModelResolver 
from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from fraud_detection.entity import config_entity,artifact_entity 
from fraud_detection.utils import load_object
from fraud_detection.config import TARGET_COLUMN
import pandas as pd 
import os,sys 
from sklearn.metrics import f1_score 



class ModelEvaluation:
    def __init__(self,model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact
                 ):
                 try:
                    logging.info(f"{'<<' *20}  MODEL EVALUATION {'>>' *20}")
                    self.model_eval_config=model_eval_config
                    self.data_ingestion_artifact=data_ingestion_artifact
                    self.data_transformation_artifact=data_transformation_artifact 
                    self.model_trainer_artifact=model_trainer_artifact
                    self.model_resolver=ModelResolver()

                 except Exception as e:
                    raise FraudException(e,sys)  

    
    def initiate_model_evaluation(self):
        try:
            latest_dir_path=self.model_resolver.get_latest_dir_path() 
            if latest_dir_path ==None:
                model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact
            

            logging.info(f"finding the location of transformer path and model path")
            transformer_path=self.model_resolver.get_latest_transformer_path() 
            model_path=self.model_resolver.get_latest_model_path()

            
            logging.info(f"previous trained objects of transformar and model")
            transformer=load_object(file_path=transformer_path)
            model=load_object(file_path=model_path)


            logging.info(f"currently trained transformation and model objects")
            current_transformer=load_object(file_path=self.data_transformation_artifact.transform_object_path) 
            current_model=load_object(file_path=self.model_trainer_artifact.model_path)


            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df=test_df[TARGET_COLUMN]
            y_true=target_df 

            input_feature_name=list(transformer.feature_names_in_)
            input_arr=transformer.transform(test_df[input_feature_name])   
            y_pred=model.predict(input_arr)
            previous_model_score=f1_score(y_true=y_true,y_pred=y_pred)  
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")


            input_feature_name=list(current_transformer.feature_names_in_)
            input_arr=current_transformer.transform(test_df[input_feature_name])
            y_pred2=current_model.predict(input_arr)
            current_model__score=f1_score(y_true=y_true,y_pred=y_pred2)
            logging.info(f"Accuracy using current trained model: {current_model__score}")


            if current_model__score<previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception(f"current model is not better thanb previous model")

            
            logging.info(f"preparing model evaluation artifact")
            model_eval_artifact=artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                           improved_accuracy=current_model__score-previous_model_score)

            logging.info(f"model_evaluation_artifact: {model_eval_artifact}")
            return model_eval_artifact

            
        except Exception as e:
            raise FraudException(e,sys)