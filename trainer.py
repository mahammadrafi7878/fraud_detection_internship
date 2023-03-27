from fraud_detection.pipeline.training_pipeline import start_training_pipeline


file_path="/config/workspace/large_fraud_data11.csv"
print(__name__)
if __name__=="__main__":
    try:
        start_training_pipeline()
    except Exception as e:
        raise FraudException(e,sys)