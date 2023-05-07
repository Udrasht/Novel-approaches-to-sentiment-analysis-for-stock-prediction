from sklearn.ensemble import AdaBoostClassifier

def AdaBoost(x_train, x_test, y_train, y_test ):
  model = AdaBoostClassifier(n_estimators=150, learning_rate=0.2)
  model.fit(x_train, y_train)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)
  return model, y_pred_train, y_pred_test



