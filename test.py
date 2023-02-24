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


print(folder)