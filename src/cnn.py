from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np


def CNN(x_train, x_test, y_train, y_test ):
    m1, n1 = x_train.shape
    x_train = np.array(x_train).reshape(m1, n1, 1)
    m2, n2 = x_test.shape
    x_test = np.array(x_test).reshape(m2, n2, 1)
    
    batch_size = 64
    epochs = 200
    
    model = Sequential()
    model.add( Conv1D(64, 3, activation='relu', input_shape = x_train.shape[1:3] ))
    
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    
    # model.add(Conv1D(256, 3, activation='relu'))
    # model.add(MaxPooling1D(3))
    
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.9))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.9))
    
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train model
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1 )
    
    y_pred_train = model.predict(x_train)
    y_pred_train = (y_pred_train > 0.5)
    
    y_pred_test = model.predict(x_test)
    y_pred_test = (y_pred_test > 0.5)

    return model, y_pred_train, y_pred_test



  
