import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from textblob import TextBlob
import re


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)


def processText(column,n):#to remove stop words,to convert into lowercase and root word
    corpus = []
    for i in range(n):
        review = re.sub('[^a-zA-Z]', ' ', column[i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus



def embed(input):
  return model(input)


def find_sentment(corpus):
  sentimentColumns=[]
  n=0
  p=0
  for content in corpus:
    blob = TextBlob(content)
    sentiment=blob.sentiment.polarity
    if sentiment<0:
      sentimentColumns.append(0)
      n=n+1
    else:
      sentimentColumns.append(1)
      p=p+1
  print(n,p,"sentement")
  return sentimentColumns


def normalize_data(df):
  scaler = MinMaxScaler(feature_range=(0, 1))
  df = scaler.fit_transform(df)
  return df


def apply_PCA(titleColumn, field="t", n_comp_pca = 30 ):
  text_vector=[]
  for text in titleColumn:
      sentence = text
      message_embeddings = embed([sentence])
      temp=np.array(message_embeddings[0])
      temp=temp.T
      text_vector.append(temp)
  text_vector=np.array(text_vector)
  text_vector = normalize_data(text_vector)
  pca = PCA(n_components = n_comp_pca)
  pca.fit(text_vector)
  data_pca = pca.transform(text_vector)
  cols = [ 'pca_'+str(i)+'_'+field for i in range(1,n_comp_pca+1) ]
  df_after_pca=pd.DataFrame(data_pca, columns = cols)
  return df_after_pca


  
def concat_data(df,df_after_pca,dateColumn,sentement_title,df_after_pca_summery,sentementvalue_of_summary, orig_col, run_mode):
  if( run_mode == 2 ):
      df=pd.DataFrame(df,columns=orig_col)
      new_data=pd.concat([dateColumn,df,df_after_pca,df_after_pca_summery],axis=1)
      new_data['sentiment_title'] = sentement_title
      new_data['sentiment_summary'] = sentementvalue_of_summary
  elif( run_mode == 0 ):
      new_data=pd.concat([dateColumn,df_after_pca,df_after_pca_summery],axis=1)
      new_data['sentiment_title'] = sentement_title
      new_data['sentiment_summary'] = sentementvalue_of_summary
  else:
      df=pd.DataFrame(df,columns=orig_col)
      new_data=pd.concat([dateColumn,df],axis=1)  
  return new_data


def save_files( ticker, processed_data_dict, run_mode ):
    if( run_mode == 0 ):
        folder_name = "processed_news_data"
    elif( run_mode == 1 ):
        folder_name = 'processed_numerical_data'
    else:
        folder_name = 'processed_data'
    filename = ticker + "_processed.csv" 
    data_path = folder_name + "\\" + filename
    processed_data_dict[ticker].to_csv( data_path )
    print( filename,'Saved Successfully' )
    

# Set the directory path
dir_path = 'data/'

# Filter only CSV files
all_tickers = ["INTU", "PYPL", "ADBE", "ORCL", "EBAY", "AMZN", "NFLX", "GM", "AAPL", "MSFT" ]

# RUN MODE
# 0 -> only news
# 1 -> only numeric data 
# 2 -> combine 

run_mode = 2

processed_data_dict = {}
  
for ticker in all_tickers:
    #ticker = "AAPL"
    df = pd.read_csv( dir_path + "\\"+ ticker + "_combine.csv"  )
    df = df.astype( {'summary':'string'} )
    df = df.astype( {'title':'string'} )
    # df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y')
    
    titleColumn = df['title']
    dateColumn = df['datetime']
    summaryColumn = df['summary']
    
    titleColumn1 = titleColumn
    summaryColumn1 = summaryColumn
    
    #preprocessing title column
    titleColumn = processText(titleColumn,len(titleColumn))
    sentement_title = find_sentment(titleColumn)

    #preprocessing summary column
    summaryColumn = processText(summaryColumn,len(summaryColumn))
    sentement_summary = find_sentment(summaryColumn)
    

    df=df.drop(['datetime','title','summary','y_actual'],axis='columns')
    orig_col = df.columns
    df = normalize_data(df)

    df_after_pca = apply_PCA( titleColumn1, "t")
    df_after_pca_summery = apply_PCA( summaryColumn1, "s")
   
    #concate news pca data with the ofiginal data
    new_data = concat_data(df,df_after_pca,dateColumn,sentement_title,df_after_pca_summery,sentement_summary, orig_col, run_mode )
    
    #divide data in test train
    new_data = new_data.drop('datetime', axis=1)
    
    #ticker = item.split('_')[0]
    processed_data_dict[ticker] = new_data
    save_files( ticker, processed_data_dict, run_mode )







