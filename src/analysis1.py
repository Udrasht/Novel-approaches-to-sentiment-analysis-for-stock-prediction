import pandas as pd
 
def procss( file_name ):
    df_combine = pd.read_excel( file_name, sheet_name='Testing Accuracy' )
    df_combine = df_combine.rename( { df_combine.columns[0]: 'models' }, axis=1 )
    df_combine['mean_acc'] = df_combine.mean(axis=1)
    return df_combine

df_combine = procss(  file_name = 'metrics.xls' )
df_news = procss(  file_name = 'metrics_news.xls' )
df_numerical = procss(  file_name = 'metrics_numerical.xls' )

df_final_acc = pd.DataFrame()
df_final_acc['models'] = df_combine['models']
df_final_acc['combine_acc'] = df_combine['mean_acc']
df_final_acc['news_acc'] = df_news['mean_acc']
df_final_acc['numerical_acc'] = df_numerical['mean_acc']
df_final_acc.to_csv('final_mean_metrics.csv')


df_combine1 = df_combine.drop(["PYPL","ADBE","AMZN","INTU","ORCL"], axis = 1)
df_news1 = df_news.drop(["PYPL","ADBE","AMZN","INTU","ORCL"], axis = 1)
df_numerical1 = df_numerical.drop(["PYPL","ADBE","AMZN","INTU","ORCL"], axis = 1)

df_news1['mean_acc'] = df_news1.iloc[:,:-1].mean(axis=1)
df_combine1['mean_acc'] = df_combine1.iloc[:,:-1].mean(axis=1)
df_numerical1['mean_acc'] = df_numerical1.iloc[:,:-1].mean(axis=1)


df_final_acc1 = pd.DataFrame()
df_final_acc1['models'] = df_combine1['models']
df_final_acc1['combine_acc'] = df_combine1['mean_acc']
df_final_acc1['news_acc'] = df_news1['mean_acc']
df_final_acc1['numerical_acc'] = df_numerical1['mean_acc']
df_final_acc1.to_csv('final_mean_metrics1.csv')


