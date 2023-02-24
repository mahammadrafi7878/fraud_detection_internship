import pymongo
import pandas as pd
import json

client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATABASE_NAME='fraud_detection'
COLLECTION_NAME="predicting_fraud"
DATA_FILE_PATH= "/config/workspace/total_data.csv"


def drop_index_column(data):
    if 'Unnamed: 0.1' in data.columns:
        data.drop('Unnamed: 0.1',axis=1,inplace=True)
    if  'Unnamed: 0' in data.columns:
        data.drop( 'Unnamed: 0',axis=1,inplace=True)

    return data


df=pd.read_csv(DATA_FILE_PATH)
print(df.columns)

drop_index_column(df)

print(f"df.columns after droping index column {df.columns}")

json_records=list(json.loads(df.T.to_json()).values())
print(json_records[0])

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)