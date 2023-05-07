from sklearn import svm

def SVM(x_train, x_test, y_train, y_test ):
  model = svm.SVC(C=3, kernel='rbf',max_iter=100000)
  model.fit(x_train, y_train)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)
  return model, y_pred_train, y_pred_test
