from sklearn.ensemble import RandomForestClassifier

def RandomForest(x_train, x_test, y_train, y_test ):
  model = RandomForestClassifier(n_estimators=12, oob_score=True, random_state=0)
  model.fit(x_train, y_train)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)
  return model, y_pred_train, y_pred_test





