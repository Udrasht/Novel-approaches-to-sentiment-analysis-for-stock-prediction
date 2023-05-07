from keras.models import Sequential
from keras.layers import LSTM, Dense 
import numpy as np
  
    
def RNN(x_train,x_test,y_train,y_test, batch_size = 128, epochs = 20 ):
  
  m1, n1 = x_train.shape
  x_train = np.array(x_train).reshape(m1, n1, 1)

  m2, n2 = x_test.shape
  x_test= np.array(x_test).reshape(m2, n2, 1)

  model = Sequential()
  model.add(LSTM(128, dropout=0.8, recurrent_dropout=0.8,  kernel_regularizer='l2', activity_regularizer='l2', input_shape=x_train.shape[1:3]))
  model.add(Dense(100, activation='relu'))

  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
  history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)

  y_pred_train = model.predict(x_train)
  th_train = ( np.max(y_pred_train) + np.min(y_pred_train) )/2
  y_pred_train = (y_pred_train > th_train)
        
  y_pred_test = model.predict(x_test)
  th_test = ( np.max(y_pred_test) + np.min(y_pred_test) )/2
  y_pred_test = (y_pred_test > th_test)
    
  return model, y_pred_train, y_pred_test

