import pandas as pd

import os 
import re
data=os.listdir()

folder=[]
for i in data:
    result=re.findall("csv$", i)
    if result:
        folder.append(i)

if 'total_data.csv' in folder:
    folder.remove('total_data.csv')

data=[]
for file in folder:
    df=pd.read_csv(file)
    data.append(df)

total=pd.concat(data)
total.to_csv('total_data.csv')
