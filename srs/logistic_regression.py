from sklearn.linear_model import LogisticRegression

def LogisticReg(x_train, x_test, y_train, y_test):
  model = LogisticRegression(C = 1e10, tol=0.000000001, max_iter=100000)
  model.fit(x_train, y_train)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)
  return model, y_pred_train, y_pred_test





  