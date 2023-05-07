from pygooglenews import GoogleNews
import pandas as pd
from datetime import datetime, timedelta #, date
from newspaper import Article
import threading
import queue

import warnings
warnings.filterwarnings("ignore")

stock_dict = {}

def get_titles(search, from_dt, end_dt, gn ):
    stories = []
    #check = 'allintext:{}'.format(search)
    check = 'intitle:{}'.format(search)

    # search=gn.search(check,when='48m',n=1000,from_='2018-07-01',to_ ='2022-07-01')
    # search=gn.search('intitle:{}'.format(search),when='48m')
    # search=gn.search(search,when='15d')
    # search = gn.search( search, from_ = from_dt, to_ = end_dt )
    search = gn.search( check, from_ = from_dt, to_ = end_dt )
    newsitem = search['entries']
    for item in newsitem:
        story = {
            'date': item.published,
            'title': item.title,
            'link':item.link
        }
        stories.append(story)
    return stories


def convert_to_datetime(a):
    return datetime.strptime(a, "%a, %d %b %Y %H:%M:%S %Z").date()

# using multi-threading
def fetch_news_helper3( ticker, gn, time_delta_orig, start_dt_str, till_date, stocks_queries ):
    global stock_dict
    stock_q = stocks_queries[ticker]
    
    final_df = pd.DataFrame(columns = ['datetime','title','link'] )
    
    from_dt_str = start_dt_str
    from_dt = datetime.strptime(from_dt_str, "%Y-%m-%d")
    print(ticker, from_dt)
    
    time_delta = time_delta_orig
    
    while( from_dt.date() <= till_date ): 
        
        diff_days = (till_date - from_dt.date()).days
        if( diff_days < time_delta ):
            time_delta = diff_days
        
        end_dt = from_dt + timedelta(days = time_delta)
        #end_dt_str = end_dt.strftime('%Y-%m-%d')
        end_dt1 = end_dt + timedelta(days = 1)
        end_dt1_str = end_dt1.strftime('%Y-%m-%d')
        
        print( from_dt_str, 'to', end_dt1_str, "(end date is exclusive)" )
        
        stories = get_titles( stock_q, from_dt_str, end_dt1_str, gn )
        df = pd.DataFrame(stories)
        
        if( len(df) > 0 ):
            df['datetime'] = df.date.apply( convert_to_datetime )
            df = df.sort_values(by='datetime').reset_index(drop=True)
            df = df.drop(['date'],axis=1)
            #df = df.set_index('datetime')
            df = df[ df.index != end_dt1.date() ]
        
        final_df = pd.concat([final_df, df], axis=0)
        #df1 = pd.concat([final_df, df], axis=0)
        
        from_dt_str = end_dt1_str
        from_dt = datetime.strptime(from_dt_str, "%Y-%m-%d")
        
    final_df1 = pd.DataFrame( final_df.groupby('datetime')['title'].apply(list) )
    final_df1['link'] = final_df.groupby('datetime')['link'].apply(list) 
    stock_dict[ticker] = final_df1
    print()

# using multi-threading
def fetch_news_helper2(  gn, all_tickers, time_delta_orig, start_dt_str, till_date, stocks_queries ):
    global stock_dict
    
    threads = []
    for ticker in all_tickers:
        t = threading.Thread(target=fetch_news_helper3, args=(ticker, gn, time_delta_orig, start_dt_str, till_date, stocks_queries) )
        threads.append(t)
        t.start()
        print('Started: %s' % t)
        
    for t in threads:
        t.join()
        
    return stock_dict

# sequentially
def fetch_news_helper(  gn, all_tickers, time_delta_orig, start_dt_str, till_date, stocks_queries ):
    stock_dict = {}
    for ticker in all_tickers:
        #ticker = 'AAPL'
        stock_q = stocks_queries[ticker]
        
        final_df = pd.DataFrame(columns = ['datetime','title','link'] )
        
        from_dt_str = start_dt_str
        from_dt = datetime.strptime(from_dt_str, "%Y-%m-%d")
        print(ticker, from_dt)
        
        time_delta = time_delta_orig
        
        while( from_dt.date() <= till_date ): 
            
            diff_days = (till_date - from_dt.date()).days
            if( diff_days < time_delta ):
                time_delta = diff_days
            
            end_dt = from_dt + timedelta(days = time_delta)
            #end_dt_str = end_dt.strftime('%Y-%m-%d')
            end_dt1 = end_dt + timedelta(days = 1)
            end_dt1_str = end_dt1.strftime('%Y-%m-%d')
            
            print( from_dt_str, 'to', end_dt1_str, "(end date is exclusive)" )
            
            stories = get_titles( stock_q, from_dt_str, end_dt1_str, gn )
            df = pd.DataFrame(stories)
            
            if( len(df) > 0 ):
                df['datetime'] = df.date.apply( convert_to_datetime )
                df = df.sort_values(by='datetime').reset_index(drop=True)
                df = df.drop(['date'],axis=1)
                #df = df.set_index('datetime')
                df = df[ df.index != end_dt1.date() ]
            
            final_df = pd.concat([final_df, df], axis=0)
            #df1 = pd.concat([final_df, df], axis=0)
            
            from_dt_str = end_dt1_str
            from_dt = datetime.strptime(from_dt_str, "%Y-%m-%d")
            
        final_df1 = pd.DataFrame( final_df.groupby('datetime')['title'].apply(list) )
        final_df1['link'] = final_df.groupby('datetime')['link'].apply(list) 
        stock_dict[ticker] = final_df1
        print()
        
    return stock_dict
    

        
def fetch_news( all_tickers, start_dt_str, till_dt_str, time_delta_orig = 5):

    gn = GoogleNews()
    
    #stocks_queries = { 'AAPL':'Apple -fruit AAPL', 'MSFT':'Microsoft AND MSFT AND stock' }
    
    stocks_queries = { 'AAPL':'Apple -fruit AAPL', 'MSFT':'MICROSOFT',
                       'INTU':'INTUIT', 'PYPL':'PAYPAL', 'ADBE':'ADOBE',
                       'NVDA':'NVIDIA', 'ORCL':'ORACLE', 'EBAY':'EBAY',
                       'AMZN':'AMAZON', 'NFLX':'NETFLIX', 'GM':'GENERAL MOTORS'
                     }
    
    till_date = datetime.strptime(till_dt_str, "%Y-%m-%d").date()

    return fetch_news_helper2( gn, all_tickers, time_delta_orig, start_dt_str, till_date, stocks_queries )
    #return fetch_news_helper( gn, all_tickers, time_delta_orig, start_dt_str, till_date, stocks_queries )


def get_summary(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()    
        summ = article.summary
        if( summ == 'Javascript is DisabledYour current browser configurationis not compatible with this site.'):
            return ""
        return summ
    except:
        return ""
    

cnt1 = 0
def generate_full_summary(day, lst):
    global summary_dict, cnt1
    new_lst = []
    cnt2 = 0
    for url in lst:
        summ = get_summary(url)
        if( summ != "" ):
            cnt2 += 1
        new_lst.append( summ )
    summary_dict[day] = new_lst
    cnt1 += 1
    print( cnt1,').' ,'Day', day,':', cnt2,'/',len(lst) )



all_tickers = ["INTU", "PYPL", "ADBE", "ORCL", "EBAY", "AMZN", "NFLX", "GM","MSFT","AAPL"]

beg_date = '2019-01-01'
end_date = '2022-12-31'
no_of_threads = 50          # for generating summary


stock_news_dict = fetch_news( all_tickers, start_dt_str = beg_date, till_dt_str = end_date )
    

####################   Extracting Summary via Multi-Threading    ##########################


def func(lst, thread_count = 15):
    q = queue.Queue()
    def worker():
        while True:
            y = q.get()
            if y is None:
                break
            day,x = y
            #do_sth(x)
            generate_full_summary(day,x)
            q.task_done()

    threads = []
    
    for x in range(0, thread_count):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        print('Started: %s' % t)
    
    i = 1
    for x in lst:
        q.put( (i,x) )
        i += 1
    
    # block until all tasks are done
    q.join()
    
    # stop workers
    for _ in threads:
        q.put(None)
    
    for t in threads:
        t.join()
        


def save_files( ticker, stock_news_dict ):
    folder_name = 'data'
    filename = ticker + "_news.csv" 
    data_path = folder_name + "\\" + filename
    stock_news_dict[ticker].to_csv( data_path )
    print( filename,'Saved Successfully' )


for ticker in all_tickers:
    print(ticker,"-------------")
    summary_dict = dict()
    lst = list(stock_news_dict[ticker]['link'])
    func(lst, thread_count = no_of_threads)
    
    final_lst = []
    for day, summ in sorted(summary_dict.items()):
        final_lst.append(summ)
    
    stock_news_dict[ticker]['summary'] = final_lst
    save_files( ticker, stock_news_dict )
    

####################################################








