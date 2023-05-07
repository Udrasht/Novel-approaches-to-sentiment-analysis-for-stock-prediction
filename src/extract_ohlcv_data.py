import pandas as pd
from yahoofinancials import YahooFinancials
import datetime as dt
import sys
sys.path.append('F:/DO NOT DELETE THIS FOLDER/Desktop/smai_project/stock_prediction_project/OurProject')
import indicators as ind
import numpy as np
import ast

def add_indicators( DF ):
    df = DF.copy()
    #ticker = "AAPL"
    #df = ohlc_dict[ticker].copy()
    df = ind.EVM(df)
    df = ind.ForceIndex(df)
    df['rsi'] = ind.RSI(df)
    df['cci'] = ind.CCI(df)
    df[['macd','signal']] = ind.MACD(df)
    df['macd-signal'] = df['macd'] - df['signal']
    df.drop(['macd','signal'], inplace=True, axis=1)
    df = ind.ATR(df)
    df = ind.VWAP(df)
    df = ind.SuperTrend(df)
    df = df.dropna()
    df = df.reset_index(drop=True)
    #df = ind.BollBand(df)
    return df


def combine_lst( list_of_lists ):
    return sum(list_of_lists, [])


def combine_ohlc_news( ohlc_df, news_df ):
    df11 = ohlc_df.copy()
    df12 = news_df.copy()
    df11['datetime1'] = df11.datetime
    df11.datetime = pd.to_datetime(df11.datetime)
    df12.datetime = pd.to_datetime(df12.datetime)
    #df12 = df12.drop(3,axis = 0)
    df13 = pd.merge(df11, df12 ,on='datetime',how='outer', sort=True)
    df13.title = df13.title.ffill() 
    df13.summary = df13.summary.ffill() 
    df13.datetime1 = df13.datetime1.bfill() 
    df14 = df13[ ~df13.datetime1.isnull() ]
    df15 = df14.groupby('datetime1')['title'].apply(combine_lst)
    df15 = df15.reset_index()
    df14 = df14[ df14.close > 0 ]
    df16 = pd.merge(df14, df15 ,on='datetime1',how='inner')
    df16 = df16.drop(['title_x','datetime'], axis = 1)
    df16 = df16.rename({ 'title_y':'title', 'datetime1':'datetime' }, axis=1) 
    df16 = df16.set_index('datetime')
    return df16


def combine_ohlc_financials(  ohlc_df, fin_df ):
    ohlc_df['year'] = pd.to_datetime(ohlc_df.datetime).dt.year
    ohlc_fin_df = pd.merge(ohlc_df, fin_df ,on='year',how='left')
    ohlc_fin_df = ohlc_fin_df.drop('year',axis=1)
    return ohlc_fin_df



all_tickers = ["INTU", "PYPL", "NVDA", "ORCL", "EBAY", "AMZN", "NFLX", "GM"]

beg_date = '2019-01-01'
end_date = '2022-12-31'

folder_name = 'data'

ohlc_dict = {}
stock_news_dict = {}
financials_dict = {}

for ticker in all_tickers:
    filename = ticker + "_news.csv" 
    data_path = folder_name + "\\" + filename
    stock_news_dict[ticker] = pd.read_csv(data_path).set_index('datetime')
    stock_news_dict[ticker]['title'] = stock_news_dict[ticker]['title'].apply(lambda x: ast.literal_eval(x)) 
    stock_news_dict[ticker]['summary'] = stock_news_dict[ticker]['summary'].apply(lambda x: ast.literal_eval(x))
    stock_news_dict[ticker].drop(['link'], inplace=True, axis=1)
    
for ticker in all_tickers:
    filename = ticker + "_financials.csv" 
    data_path = folder_name + "\\" + filename
    financials_dict[ticker] = pd.read_csv(data_path)
        

combined_data_dict = {}


def save_files( ticker, combined_data_dict ):
    folder_name = 'data'
    filename = ticker + "_combine.csv" 
    data_path = folder_name + "\\" + filename
    combined_data_dict[ticker].to_csv( data_path )
    print( filename,'Saved Successfully' )

for ticker in all_tickers:
    yahoo_financials = YahooFinancials(ticker)
    json_obj = yahoo_financials.get_historical_price_data(beg_date,end_date,"daily")
    ohlv = json_obj[ticker]['prices']
    temp = pd.DataFrame(ohlv)[["formatted_date","adjclose","open","low","high","volume"]]
    temp.rename( { "adjclose":"close", "formatted_date":"datetime" }, inplace = True, axis = 1 )
    #temp.set_index( "datetime",inplace=True )
    #temp.dropna(inplace=True)
    temp['y_actual'] = np.where( temp['close'].shift(-1) > temp['close'] , 1, 0 )
    temp.drop(temp.tail(1).index,inplace=True)    #drops last row of data frame

    ohlc_dict[ticker] = add_indicators( temp )
    ohlc_fin_df = combine_ohlc_financials( ohlc_dict[ticker].copy(), financials_dict[ticker].copy() )
    #ohlc_dict[ticker]['news'] = stock_news_dict[ticker]
    combined_data_dict[ticker] = combine_ohlc_news( ohlc_fin_df, stock_news_dict[ticker].reset_index()  )
    save_files( ticker, combined_data_dict )

