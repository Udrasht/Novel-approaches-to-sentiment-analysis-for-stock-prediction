import requests
from bs4 import BeautifulSoup
import pandas as pd


def store_financials(tickers):
    income_statatement_dict = {}
    balance_sheet_dict = {}
    cashflow_st_dict = {}
    
    for ticker in tickers:
        #scraping income statement
        url = "https://finance.yahoo.com/quote/{}/financials?p={}".format(ticker,ticker)
        income_statement = {}
        table_title = {}
        
        headers = {"User-Agent" : "Chrome/96.0.4664.110"}
        page = requests.get(url, headers=headers)
        page_content = page.content
        soup = BeautifulSoup(page_content,"html.parser")
        tabl = soup.find_all("div" , {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
        for t in tabl:
            heading = t.find_all("div" , {"class": "D(tbr) C($primaryColor)"})
            for top_row in heading:
                table_title[top_row.get_text(separator="|").split("|")[0]] = top_row.get_text(separator="|").split("|")[1:]
            rows = t.find_all("div" , {"class": "D(tbr) fi-row Bgc($hoverBgColor):h"})
            for row in rows:
                income_statement[row.get_text(separator="|").split("|")[0]] = row.get_text(separator="|").split("|")[1:]
    
        temp = pd.DataFrame(income_statement).T
        temp.columns = table_title["Breakdown"]
        income_statatement_dict[ticker] = temp
        
        #scraping balance sheet statement
        url = "https://finance.yahoo.com/quote/{}/balance-sheet?p={}".format(ticker,ticker)
        balance_sheet = {}
        table_title = {}
        
        headers = {"User-Agent" : "Chrome/96.0.4664.110"}
        page = requests.get(url, headers=headers)
        page_content = page.content
        soup = BeautifulSoup(page_content,"html.parser")
        tabl = soup.find_all("div" , {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
        for t in tabl:
            heading = t.find_all("div" , {"class": "D(tbr) C($primaryColor)"})
            for top_row in heading:
                table_title[top_row.get_text(separator="|").split("|")[0]] = top_row.get_text(separator="|").split("|")[1:]
            rows = t.find_all("div" , {"class": "D(tbr) fi-row Bgc($hoverBgColor):h"})
            for row in rows:
                balance_sheet[row.get_text(separator="|").split("|")[0]] = row.get_text(separator="|").split("|")[1:]
    
        temp = pd.DataFrame(balance_sheet).T
        temp.columns = table_title["Breakdown"]
        balance_sheet_dict[ticker] = temp
        
        #scraping cashflow statement
        url = "https://finance.yahoo.com/quote/{}/cash-flow?p={}".format(ticker,ticker)
        cashflow_statement = {}
        table_title = {}
        
        headers = {"User-Agent" : "Chrome/96.0.4664.110"}
        page = requests.get(url, headers=headers)
        page_content = page.content
        soup = BeautifulSoup(page_content,"html.parser")
        tabl = soup.find_all("div" , {"class" : "M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)"})
        for t in tabl:
            heading = t.find_all("div" , {"class": "D(tbr) C($primaryColor)"})
            for top_row in heading:
                table_title[top_row.get_text(separator="|").split("|")[0]] = top_row.get_text(separator="|").split("|")[1:]
            rows = t.find_all("div" , {"class": "D(tbr) fi-row Bgc($hoverBgColor):h"})
            for row in rows:
                cashflow_statement[row.get_text(separator="|").split("|")[0]] = row.get_text(separator="|").split("|")[1:]
    
        temp = pd.DataFrame(cashflow_statement).T
        temp.columns = table_title["Breakdown"]
        cashflow_st_dict[ticker] = temp
        
    #converting dataframe values to numeric
    for ticker in tickers:
        for col in income_statatement_dict[ticker].columns:
            income_statatement_dict[ticker][col] = income_statatement_dict[ticker][col].str.replace(',|- ','')
            income_statatement_dict[ticker][col] = pd.to_numeric(income_statatement_dict[ticker][col], errors = 'coerce')
            cashflow_st_dict[ticker][col] = cashflow_st_dict[ticker][col].str.replace(',|- ','')
            cashflow_st_dict[ticker][col] = pd.to_numeric(cashflow_st_dict[ticker][col], errors = 'coerce') 
            if col!="ttm": #yahoo has ttm column for income statement and cashflow statement only
                balance_sheet_dict[ticker][col] = balance_sheet_dict[ticker][col].str.replace(',|- ','')
                balance_sheet_dict[ticker][col] = pd.to_numeric(balance_sheet_dict[ticker][col], errors = 'coerce')
               
    
    l1=[]
    l2=[]
    l3=[]
    
    for i in income_statatement_dict.values():
        l1.append(i)
    
    for i in cashflow_st_dict.values():
        
        l2.append(i)
    
    for i in balance_sheet_dict.values():
        l3.append(i)
    
    financials_dict = {}
    for i in range(len(l1)):
        ticker = tickers[i]
        income=l1[i].T
        cashflow=l2[i].T
        balance=l3[i].T
    
        df=pd.concat([income,cashflow,balance],axis=1)
    
        df=df.drop(['ttm'])
        df.index= pd.to_datetime(df.index)
        df = df.reset_index()
        df = df.sort_values(by='index')
        df['year'] = df['index'].dt.year
        df = df.drop(['index'],axis=1)
        df = df.set_index('year')
        
        df = df.dropna(axis=1)
        df = df[financial_cols]
        financials_dict[ticker] = df
        #print(df)
        
    return financials_dict


financial_cols = ['Total Revenue', 'Cost of Revenue', 'Gross Profit', 'Normalized EBITDA', 
'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow','Total Assets',
'Total Liabilities Net Minority Interest','Total Equity Gross Minority Interest', 'Total Capitalization' ]   
    
#all_tickers = ["ADBE"]

all_tickers = ["INTU", "PYPL", "ADBE", "ORCL", "EBAY", "AMZN", "NFLX", "GM", "MSFT", "AAPL" ]
financials_dict = store_financials(all_tickers)

folder_name = 'data'
for ticker in all_tickers:
    filename = ticker + "_financials.csv" 
    data_path = folder_name + "\\" + filename
    financials_dict[ticker].to_csv(data_path)
    print(filename,'Saved Successfully')

    
