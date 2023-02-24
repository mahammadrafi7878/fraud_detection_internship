import pandas as pd
folder=['/config/workspace/fraud_detection1.csv',
      '/config/workspace/fraud_detection2.csv',
      '/config/workspace/fraud_detection3.csv',
      '/config/workspace/fraud_detection4.csv',
      '/config/workspace/fraud_detection5.csv',
      '/config/workspace/fraud_detection6.csv',
      '/config/workspace/fraud_detection7.csv',
      '/config/workspace/fraud_detection8.csv',
      '/config/workspace/fraud_detection9.csv'] 


data=[]
for file in folder:
    df=pd.read_csv(file)
    data.append(df)

total=pd.concat(data)
total.to_csv('total_data.csv')
