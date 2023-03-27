from fraud_detection.pipeline.training_pipeline import start_training_pipeline
from fraud_detection.pipeline.batch_predictor import satrt_batch_prediction 


file_path='/config/workspace/large_fraud_data11.csv'

print(__name__)
if __name__=="__main__":
     try:
          #start_training_pipeline()
          output_file = satrt_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e:
          raise FraudException(e,sys)
    


