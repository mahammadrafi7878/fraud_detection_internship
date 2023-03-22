import pandas as pd
import numpy as ns 
from datetime import datetime

TX_DATETIME='TX_DATETIME'


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
        df['TX_DURING_NIGHT']=is_night(df,TX_DATETIME) 
        return df 
    except Exception as e:
        raise FraudException(e,sys)



        
def get_customer_spending_behaviour_features(customer_transactions, windows_sizes_in_days=[1,7,30]):
    try:
        customer_transactions=customer_transactions.sort_values('TX_DATETIME')
        customer_transactions.index=customer_transactions.TX_DATETIME
        for window_size in windows_sizes_in_days:
            SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
            NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()
            AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW
            customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
        customer_transactions.index=customer_transactions.TRANSACTION_ID
        return customer_transactions

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

 
