import logistic_regression as lr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import build_metrics_sheet as save_metrics
import pandas as pd


def divide_data( x_data, y_data, test_size = 0.25 ):
  n = len(x_data)
  t = int(test_size*n)
  x_test = x_data.iloc[-t:,:]
  y_test = y_data.iloc[-t:]
  x_train = x_data.iloc[:t,:]
  y_train = y_data.iloc[:t]
  return x_train,x_test,y_train,y_test  


def ML_Model( ticker, x_train, x_test, y_train, y_test, algo ):
  # x_train['Volume'] = np.log(x_train['Volume']+0.5)
  # x_test['Volume'] = np.log(x_test['Volume']+0.5)
  if algo == "Logistic Regression":
      model, y_pred_train, y_pred_test = lr.LogisticReg(x_train, x_test, y_train, y_test)  
      
  metrics_utility( ticker, model, x_train, x_test, y_train, y_test, y_pred_train, y_pred_test, algo)
    

def print_performance_metrics(model, y_test, y_pred, acc_train, acc_test, f1_train, f1_test ):
  roc_auc = round(roc_auc_score(y_test, y_pred),3)
  print('Train Acc: ', acc_train, " | Test Acc:", acc_test )
  print('Train F1 Score: ', f1_train, " | Test F1 Score:", f1_test )
  print('ROC AUC Score:', roc_auc )
  print()
  print('\n clasification report:\n', classification_report(y_test, y_pred))
  print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))
  

def metrics_utility( ticker, model, x_train, x_test, y_train, y_test, y_pred_train, y_pred_test, algo):
  print("traing r2_score : %.3f" % model.score(x_train, y_train))
  print("test r2_score: %.3f" % model.score(x_test, y_test))
  #print('Coefficients: \n', baseline_LR.coef_)
  acc_train = accuracy_score(y_train, y_pred_train)
  acc_test = accuracy_score(y_test, y_pred_test)
  
  f1_train = f1_score(y_train, y_pred_train, average='weighted')
  f1_test = f1_score(y_test, y_pred_test, average='weighted')
  
  acc_train = round(acc_train,3)
  acc_test = round(acc_test, 3)
  f1_train = round(f1_train,3)
  f1_test = round(f1_test,3)
  print_performance_metrics(model, y_test, y_pred_test, acc_train, acc_test, f1_train, f1_test )
  #ticker = item.split('_')[0]
  save_metrics.save_metrics_in_excel( algo = algo, ticker = ticker,  train_acc = acc_train, test_acc = acc_test, f1_train = f1_train, f1_test = f1_test  )



data_folder = 'data'
folder_name = 'processed_data'
all_tickers = ["INTU", "PYPL", "ADBE", "ORCL", "EBAY", "AMZN", "NFLX", "GM", "AAPL", "MSFT" ]

for ticker in all_tickers:
    #ticker = "AMZN"
    filename1 = ticker + "_processed.csv" 
    data_path = folder_name + "\\" + filename1
    new_data = pd.read_csv(data_path, index_col  = 0  )
    
    filename2 = ticker + "_combine.csv" 
    df = pd.read_csv(data_folder + "\\" + filename2)
    y_actual = df['y_actual']
    
    x_train, x_test, y_train, y_test = divide_data( new_data, y_actual, test_size = 0.25 )
    ML_Model( ticker, x_train, x_test, y_train, y_test, 'Logistic Regression') #Apply Model

