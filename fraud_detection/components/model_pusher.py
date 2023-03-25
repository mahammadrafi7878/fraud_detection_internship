from fraud_detection.exception import FraudException 
from fraud_detection.logger import logging 
from fraud_detection.predictor import ModelResolver 
from fraud_detection.entity.config_entity import ModelPusherConfig 
import os,sys 
from fraud_detection.utils import save_object,load_object 
from fraud_detection.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact 


class ModelPusher:
    def __init__(self,
                 model_pusher_config:ModelPusherConfig,
                 model_trainer_artifact:ModelTrainerArtifact,
                 data_transformation_artifact:DataTransformationArtifact,

                 
                ):

                try:
                    logging.info(f"{'<<' *20}  MODEL PUSHER {'>>' *20}") 
                    
                    self.model_pusher_config=model_pusher_config
                    self.model_trainer_artifact=model_trainer_artifact
                    self.data_transformation_artifact=data_transformation_artifact 
                    self.model_resolver = ModelResolver(model_registery=self.model_pusher_config.saved_model_dir)
                except Exception as e:
                    raise FraudException(e,sys)

    def initiate_model_pusher(self):
        try:
            logging.info(f"loading transformer and model")
            transfomer=load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model=load_object(file_path=self.model_trainer_artifact.model_path)

            logging.info(f"saving the model into pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transfomer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)


            logging.info(f"saving the model in saved model directory")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()

            
            save_object(file_path=transformer_path, obj=transfomer)
            save_object(file_path=model_path, obj=model)
            
            logging.info(f"preparing model pusher artifact")
            model_pusher_artifact=ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
             saved_model_dir=self.model_pusher_config.saved_model_dir)
            
            logging.info(f"model pusher artifact : {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise FraudException(e, sys)