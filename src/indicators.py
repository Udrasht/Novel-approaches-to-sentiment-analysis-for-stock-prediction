import numpy as np
import pandas as pd
#import json
import pandas_ta as ta
from ta.volume import VolumeWeightedAveragePrice

# Ease of Movement
def EVM(data, ndays = 14):
    dm = ((data['high'] + data['low'])/2) - ((data['high'].shift(1) + data['low'].shift(1))/2)
    br = (data['volume'] / 100000000) / ((data['high'] - data['low']))
    EVM = dm / br
    EVM_MA = pd.Series( EVM.rolling(ndays).mean(), name = 'evm')
    data = data.join(EVM_MA)
    return data

# Force Index
def ForceIndex(data, ndays = 14):
    FI = pd.Series(data['close'].diff(ndays) * data['volume'], name = 'force_index')
    data = data.join(FI)
    return data


def RSI(df, n=14):
    "function to calculate RSI"
    delta = df["close"].diff().dropna()
    u = delta * 0
    if(len(u) < n):
        rsi = pd.Series([np.nan]*len(df))
        return rsi
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean( u[:n] ) #first value is sum of avg gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean( d[:n] ) #first value is sum of avg losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com = n-1,adjust = False).mean()/d.ewm(com = n-1,adjust = False).mean()
    rsi = 100 - 100 / (1 + rs)
    return rsi

def MACD(DF,a=12,b=26,c=9):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["close"].ewm(span=b,min_periods=b).mean()
    df["macd"]=df["MA_Fast"]-df["MA_Slow"]
    df["signal"]=df["macd"].ewm(span=c,min_periods=c).mean()
    #df.dropna(inplace=True)
    return df[["macd","signal"]]

def CCI(DF, n=14): 
    data = DF.copy()
    CCI = ta.cci(high=data['high'], low=data['low'], close=data["close"], length = n)
    return CCI


def ATR1(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    atr = 'ATR_' + str(period)
    if not 'TR1' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())         
        df['TR1'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)         
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)
    EMA(df, 'TR1', atr, period, alpha=True)    
    return df

def EMA(df, base, target, period, alpha=False):
    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])    
    if (alpha == True):
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        df[target] = con.ewm(span=period, adjust=False).mean()
        df[target].fillna(0, inplace=True)
    return df

def SuperTrend(DF, period = 7, multiplier = 3, ohlc=['open', 'high', 'low', 'close']):
    df = DF.copy()
    ATR1(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'st_' + str(period) + '_' + str(multiplier)
    stx = 'stx_' + str(period) + '_' + str(multiplier)
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] <  df['final_lb'].iat[i] else 0.00 
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down',  'up'), np.NaN)
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    #df.fillna(0, inplace=True)
    df.drop(['TR1',atr,st], inplace=True, axis=1)
    df[stx] = np.where( df[stx] == 'up' , 1, -1 )
    return df    

def ATR(DF,n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['high']-df['low'])
    df['H-PC']=abs(df['high']-df['close'].shift(1))
    df['L-PC']=abs(df['low']-df['close'].shift(1))
    df['tr']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['atr'] = df['tr'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC','tr'],axis=1)
    return df2

def BollBand(DF,n=20,std=2):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["bb_mid"] = df['close'].rolling(n).mean()
    df["bb_up"] = df["bb_mid"] + std*df['close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["bb_dn"] = df["bb_mid"] - std*df['close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    #df["BB_width"] = df["BB_up"] - df["BB_dn"]
    #df.dropna(inplace=True)
    return df


def VWAP( df, length=14, fillna=True):
    df['vwap'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df["close"], volume=df['volume'], window=length, fillna=fillna).volume_weighted_average_price()
    return df

