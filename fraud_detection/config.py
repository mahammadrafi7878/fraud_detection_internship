import os,sys
import pymongo
from dataclasses import dataclass 
from dotenv import load_dotenv
print("Dot env variable loading")
load_dotenv 

@dataclass

class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")

object=EnvironmentVariable()
mongo_db=pymongo.MongoClient(object.mongo_db_url)  


TARGET_COLUMN='NEW_TX_FRAUD'
