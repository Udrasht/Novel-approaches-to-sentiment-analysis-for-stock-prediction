from xlwt import Workbook
import xlrd
from xlutils.copy import copy
 
# Columns
all_tickers = {"AAPL":1,"MSFT":2,"INTU":3, "PYPL":4, "ADBE":5, "GM":6, "ORCL":7, "EBAY":8 , "AMZN":9, "NFLX":10}

# Rows
algorithms = {'Logistic Regression':1, 'SVM':2, 'Random Forest':3, 'MLP':4 ,'CNN':5, 'RNN':6, 'Bagging':7, 'Boosting':8}

file_name = "metrics_news.xls"

def create_excel_file():
    wb = Workbook()                   # Workbook is created
    sheet1 = wb.add_sheet('Training Accuracy')
    sheet2 = wb.add_sheet('Testing Accuracy')
    sheet3 = wb.add_sheet('Training F1 Score')
    sheet4 = wb.add_sheet('Testing F2 Score')
    
    for algos, val in algorithms.items():
        sheet1.write( val, 0, algos )
        sheet2.write( val, 0, algos )
        sheet3.write( val, 0, algos )
        sheet4.write( val, 0, algos )
       
    
    for ticker, val in all_tickers.items():
        sheet1.write( 0, val, ticker )
        sheet2.write( 0, val, ticker )
        sheet3.write( 0, val, ticker )
        sheet4.write( 0, val, ticker )
      
    wb.save(file_name)


def save_metrics_in_excel( algo, ticker, train_acc, test_acc, f1_train , f1_test ):
    row_num = algorithms[algo]
    col_num = all_tickers[ticker]
    rb = xlrd.open_workbook(file_name)       # load the excel file
    wb = copy(rb)                                 # copy the contents of excel file
    
    #if( train_acc != -1 ):
    training_sheet = wb.get_sheet(0)                # open the first sheet
    training_sheet.write( row_num, col_num, train_acc )
    
    testing_sheet = wb.get_sheet(1)               # open the second sheet
    testing_sheet.write( row_num, col_num, test_acc )
    
    f1_training_sheet = wb.get_sheet(2)               # open the second sheet
    f1_training_sheet.write( row_num, col_num, f1_train )
    
    f1_testing_sheet = wb.get_sheet(3)               # open the second sheet
    f1_testing_sheet.write( row_num, col_num, f1_test )
    
    wb.save(file_name)
    print(algo, ticker, "Updated", row_num, col_num)
    

if( __name__ == "__main__" ):
    create_excel_file()
    #save_metrics_in_excel("SVM", "MSFT", 0.99, 0.70, 0.80, 0.70 )
    
    
#import build_metrics_sheet as save_metrics
#save_metrics.save_metrics_in_excel( algo = "SVM", ticker = "AAPL",  train_acc = 0.99, test_acc = 0.70, f1_train = 0.80, f1_test = 0.7  )