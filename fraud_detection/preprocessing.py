from fraud_detection.exception import FraudException
import pandas as pd


def is_weekend(tx_datetime):
    try:
        weekday=tx_datetime.weekday()
        is_weeekend = weekday>=5
        return int(is_weeekend)
    except Exception as e:
        raise FraudException(e, sys)


def transferd_datetime(df:pd.DataFrame):
    try:
        df['TX_DURING_WEEKEND']=df['TX_DATETIME'].apply(is_weekend)
        return df 
    except Exception as e:
        raise FraudException(e, sys)


    
def is_night(df:pd.DataFrame,variable:str):
    try:
        for i in range(len(df[variable])):
            hour=df[variable][i].hour
            if hour<=6:
                return 1
            else:
                return 0
    except Exception as e:
        raise FraudException(e, sys)



def transformed_day(df:pd.DataFrame):
    try:
        df['TX_DURING_NIGHT']=is_night(df,'TX_DATETIME')
        return df 
    except Exception as e:
        raise FraudException(e, sys)    


    
def get_customer_spending_behaviour_features(customer_transactions,window_sizes_in_days=[1,7,30]):
    try:
        customer_transactions=customer_transactions.sort_values('TX_DATETIME')
        customer_transactions.index=customer_transactions.TX_DATETIME
        for window_size in window_sizes_in_days:
            SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
            NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()
        
            AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW
        
            customer_transactions['COUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
            customer_transactions['COUSTOMER_ID_AVG_AMOUNT_TX_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
        customer_transactions.index=customer_transactions.TRANSACTION_ID
    
        return customer_transactions
    except Exception as e:
        raise FraudException(e, sys)

def customer_features(df:pd.DataFrame):
    try:
        df=df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x,window_sizes_in_days=[1,7,30]))
        df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
        return df
    except Exception as e:
        raise FraudException(e, sys) 




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


def get_transaction(df:pd.DataFrame):
    try:
        df=df.groupby('TERMINAL_ID').apply(lambda x:get_count_risk_rolling_window(x, delay_period=7, windows_sizes_in_days=[1,7,30]))
        df=train_df.sort_values('TX_DATETIME').reset_index(drop=True)
        return df
    except Exception as e:
        raise FraudException(e, sys)
    
def reset_index(df:pd.DataFrame):
    try:
        df=df.reset_index(inplace=False)
        return df
    except Exception as e:
        raise FraudException(e, sys) 
