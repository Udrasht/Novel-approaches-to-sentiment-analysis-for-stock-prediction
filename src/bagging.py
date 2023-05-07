import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

itertion = 0

def Bagging(x_train,x_test,y_train,y_test):
  global itertion
  itertion = 0
  num_of_models = 7  
  rows_ratio = 0.6
  
  y_pred_test_s_lst = []
  y_pred_train_s_lst = []
  
  for i in range(num_of_models):
    n,m=x_train.shape
    n_rows = int( rows_ratio * n )
    # Generate random indices for the columns
    row_indices = np.random.choice(x_train.shape[0], n_rows, replace=False)
    # Slice the columns from X_train and y_train using the indices
    X_subset_train = x_train.iloc[row_indices]
    y_subset_train = y_train.iloc[row_indices]
   
    y_pred_test_s, y_pred_train_s  = ML_Model(x_train, X_subset_train,x_test,y_subset_train,y_test)
    y_pred_test_s_lst.append(y_pred_test_s)
    y_pred_train_s_lst.append(y_pred_train_s)
    
  
  def get_y_pred(y_pred_lst):
    y_pred_lst = np.array(y_pred_lst)
    y_pred_lst = y_pred_lst.transpose()
    num_of_1 = np.sum( y_pred_lst, axis = 1 )
    num_of_0 = num_of_models - num_of_1
    c1 = np.array(num_of_1 > num_of_0)
    overall_y_pred = c1.astype(int)    
    return overall_y_pred

  y_pred_test = get_y_pred(y_pred_test_s_lst)
  y_pred_train = get_y_pred(y_pred_train_s_lst)

  _ = None
  return _ ,  y_pred_train, y_pred_test
  
  
def ML_Model(x_train, X_train,X_test,y_train,y_test):
    learning_rates = [0.001]
    ## the optimal hidden layer size is between the input size and the output size
    hiddens = [(50, 50, 10), (50, 2), (20, 20),(10, 10),(10, 10, 10)]
    best_acc = 0

    for lrts in learning_rates :
        for hidden_layers in hiddens:
            clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hidden_layers, random_state=1, learning_rate_init = lrts, max_iter=2000)
            print ('finishing fitting...')
            clf.fit(X_train, y_train)
            (train_acc, test_acc) = evaluate(clf, X_train, y_train , X_test, y_test,lrts, True)

            if test_acc >= best_acc:
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(x_train)
                best_acc = test_acc
    return y_pred_test, y_pred_train


def get_conf_mat(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))


def evaluate(clf, X_train, y_train, X_test, y_test,lrts, verbose=True):
    global itertion
    itertion += 1
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)
    training_accu = accuracy_score(y_hat_train, y_train)
    testing_accu = accuracy_score(y_hat_test, y_test)
    
    if verbose:
        print(itertion)
        print ('The training accuracy is %f' % training_accu)
        print ('The testing accuracy is %f' % testing_accu)
        get_conf_mat(y_test, y_hat_test)
        print ('----------')
    return (training_accu, testing_accu)
  

  
  
  
