from xlwt import Workbook
import xlrd
from xlutils.copy import copy
 
# Columns
all_tickers = {"AAPL":1,"MSFT":2,"INTU":3, "PYPL":4, "ADBE":5, "NVDA":6, "ORCL":7, "EBAY":8, "AMZN":9, "NFLX":10}

# Rows
algorithms = {'Logistic Regression':1, 'SVM':2, 'Random Forest':3, 'CNN':4, 'RNN':5, 'Bagging':6, 'Boosting':7}

file_name = "accuracy.xls"

def create_excel_file():
    wb = Workbook()                   # Workbook is created
    sheet1 = wb.add_sheet('Training Accuracy')
    sheet2 = wb.add_sheet('Testing Accuracy')
    
    #for i in range( 1, len(algorithms)+1):
    for algos, val in algorithms.items():
        sheet1.write( val, 0, algos )
        sheet2.write( val, 0, algos )
    
    for ticker, val in all_tickers.items():
        sheet1.write( 0, val, ticker )
        sheet2.write( 0, val, ticker )
        
    #for i in range( 1, len(all_tickers)+1):
    #    sheet1.write(0, i, all_tickers[i-1] )
    #    sheet2.write(0, i, all_tickers[i-1] )
     
    wb.save(file_name)


def save_acc_in_excel( algo, ticker, test_acc, train_acc = -1 ):
    row_num = algorithms[algo]
    col_num = all_tickers[ticker]
    rb = xlrd.open_workbook(file_name)       # load the excel file
    wb = copy(rb)                                 # copy the contents of excel file
    testing_sheet = wb.get_sheet(1)               # open the second sheet
    testing_sheet.write( row_num, col_num, test_acc )
    if( train_acc != -1 ):
        traing_sheet = wb.get_sheet(0)                # open the first sheet
        traing_sheet.write( row_num, col_num, train_acc )
    wb.save(file_name)
    

if( __name__ == "__main__" ):
    create_excel_file()
    #save_acc_in_excel("SVM", "MSFT", 0.99, 0.70 )
    
    
#import build_accuracy_sheet as save_acc
#save_acc.save_acc_in_excel( algo = "SVM", ticker = "AAPL", test_acc = 0.70, train_acc = 0.99 )
