import pandas as pd
import numpy as ns 
from datetime import datetime
from fraud_detection.logger import logging 
from fraud_detection.exception import FraudException


TX_DATETIME='TX_DATETIME'
CUSTOMER_ID='CUSTOMER_ID'
TX_AMOUNT='TX_AMOUNT'
TERMINAL_ID='TERMINAL_ID'
TX_FRAUD='TX_FRAUD'
DELE_COLUMNS=['TX_FRAUD','TRANSACTION_ID','CUSTOMER_ID','TERMINAL_ID','TX_TIME_SECONDS','TX_TIME_DAYS','TX_FRAUD_SCENARIO']

def is_weekend(df,variable):
    try:
        for i in range(len(df[variable])):
           week=df[variable][i].weekday()
        if int(week) >5:
            return 1
        else:
            return 0
    except Exception as e:
        raise FraudException(e, sys) 

def is_week(df):
    try:
        global TX_DATETIME
        logging.info(f"Creating a feature is transaction done in week or weekend")
        df['TX_DURING_WEEKEND']=is_weekend(df,TX_DATETIME)
        return df
        
    except Exception as e:
        raise FraudException(e,sys)





        
def is_night(df,variable):
    try:
        for i in range(len(df[variable])):
            hour=df[variable][i].hour
        if int(hour)<=6:
            return 1
        else:
            return 0
    except Exception as e:
        raise FraudException(e, sys)

def is_day(df):
    try:
        global TX_DATETIME
        logging.info(f"Creating a feature with treansaction done in day or night")
        df['TX_DURING_NIGHT']=is_night(df,TX_DATETIME) 
        return df 
    except Exception as e:
        raise FraudException(e,sys)






        
def get_customer_spending_behaviour_features(customer_transactions, windows_sizes_in_days=[1,7,30]):
    try:
        #sorting values using tx_datatime
        customer_transactions=customer_transactions.sort_values('TX_DATETIME')
        customer_transactions.index=customer_transactions.TX_DATETIME
        #Iterating over windows sizes
        for window_size in windows_sizes_in_days:

            #performing some time series operation to create new features
            SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
            NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()
            AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW
            customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
        customer_transactions.index=customer_transactions.TRANSACTION_ID
        return customer_transactions

    except Exception as e:
        raise FraudException(e,sys)



def customer_feature(input_data):
    try:
        global CUSTOMER_ID
        global TX_DATETIME
        logging.info(f"Using Customer id feature creating three more features ")
        input_data=input_data.groupby(CUSTOMER_ID).apply(lambda x: get_customer_spending_behaviour_features(x,windows_sizes_in_days=[1,7,30]))
        input_data=input_data.sort_values(TX_DATETIME).reset_index(drop=True)
        return input_data
    except Exception as e:
        raise FraudException(e,sys)

    
def get_count_risk_rolling_window(terminal_transactions,delay_period,windows_sizes_in_days=[1,7,30]):
    try:
        terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')
        terminal_transactions.index=terminal_transactions.TX_DATETIME
    
        NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
        NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()
    
        for window_size in windows_sizes_in_days:
            NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
            NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()
        
            NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
            NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY 
        
        
            RISK_WINDOW=NB_FRAUD_WINDOW/NB_TX_WINDOW 
        
            terminal_transactions['TERMINAL_ID'+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            terminal_transactions['TERMINAL_ID'+'_RISK'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)
        
        terminal_transactions.index=terminal_transactions.TRANSACTION_ID
        terminal_transactions.fillna(0,inplace=True)
        return terminal_transactions


    except Exception as e:
        raise FraudException(e, sys) 


def terminal_feature(input_data):
    try:
        global TERMINAL_ID
        global TX_DATETIME

        logging.info(f"Using terminal id creating three more features")
        input_data=input_data.groupby(TERMINAL_ID).apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
        input_data=input_data.sort_values(TX_DATETIME).reset_index(drop=True) 
        return input_data 
    except Exception as e:
        raise FraudException(e,sys)




def drop_columns(input_data):
    try:
        global TX_DATETIME
        global TX_FRAUD
        global DELE_COLUMNS
        logging.info(f"After performning Feature Engineering dropping some columns")
        input_data.drop(TX_DATETIME,axis=1,inplace=True)
            
        input_data['NEW_TX_FRAUD']=input_data[TX_FRAUD].astype('int')
            
        input_data.drop(DELE_COLUMNS,axis=1,inplace=True) 
        return input_data 

        return input_data
    except Exception as e:
        raise FraudException(e,sys)


