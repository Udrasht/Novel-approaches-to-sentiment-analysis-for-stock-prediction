from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score



def NeuralNetwork(x_train, x_test, y_train, y_test, hidden_layers = (10,10,10), verbose = True ):
    model = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layers, random_state=1, max_iter=1000)
    model.fit(x_train, y_train)
    train_acc, test_acc, y_pred_train, y_pred_test = evaluate(model, x_train, y_train , x_test, y_test, verbose = verbose )
    return model, y_pred_train, y_pred_test


def evaluate( clf, x_train, y_train, x_test, y_test, verbose=True ):
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    
    training_accu = accuracy_score(y_train, y_pred_train)
    testing_accu = accuracy_score(y_test, y_pred_test)
    
    if verbose:
        print ('The training accuracy is %f' % training_accu)
        print ('The testing accuracy is %f' % testing_accu)
    return training_accu, testing_accu, y_pred_train, y_pred_test



def save_to_csv( df, ticker, algo ):
    folder_name = 'hypertuning\\' + algo
    file_name = ticker + '.csv'
    df.to_csv(folder_name + '\\' + file_name )
    print(algo,ticker,"Saved Successfully")


def NeuralNetworkOptimizer(x_train, x_test, y_train, y_test, ticker, algo = 'mlp' ):

    lst_2d = [] 
    hiddens = [(50, 50, 10), (50, 2), (20, 20),(10, 10),(10, 10, 10),(32,16,8),(20,30),(10,20,30)]

    for hid_layer in hiddens:
        best_model, y_pred_train, y_pred_test = NeuralNetwork(x_train, x_test, y_train, y_test, hidden_layers = hid_layer, verbose = False )
        #train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        lst_2d.append( [ hid_layer, test_acc ] )
    
    df = pd.DataFrame( lst_2d ,columns = [ 'hidden_layer_model', 'test_accuracy' ] )        
    save_to_csv( df, ticker, algo )
    

def divide_data( x_data, y_data, test_size = 0.25 ):
  n = len(x_data)
  t = int(test_size*n)
  x_test = x_data.iloc[-t:,:]
  y_test = y_data.iloc[-t:]
  x_train = x_data.iloc[:t,:]
  y_train = y_data.iloc[:t]
  return x_train,x_test,y_train,y_test 


if( __name__ == '__main__' ):
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
        print(ticker,"---------------")
        NeuralNetworkOptimizer(x_train, x_test, y_train, y_test, ticker )
    

