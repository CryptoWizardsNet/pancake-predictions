from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from pprint import pprint


# Run XGBOOST on algorithm
def run_xgb(target_item):
  df = pd.read_json("history.json")
  # Set targets and params
  if target_item == "bull_bear_ratio_up":
    df.loc[df["bull_ratio"].shift(-1) > df["bear_ratio"].shift(-1), "TARGET"] = 1
    df.loc[df["bull_ratio"].shift(-1) <= df["bear_ratio"].shift(-1), "TARGET"] = 0
    drop_columns = ["minute", "second", "close_price", "lock_price", "hour"]
    ne = 27
    md = 3
    lr = 0.01
    gm = 0.03
    test_size_rate = 0.3
  elif target_item == "bulls_win":
    df.loc[df["close_price"].shift(-1) > df["lock_price"].shift(-1) , "TARGET"] = 1
    df.loc[df["close_price"].shift(-1) <= df["lock_price"].shift(-1) , "TARGET"] = 0
    drop_columns = ["close_price", "lock_price"]
    ne = 19
    md = 3
    lr = 0.01
    gm = 0.03
    test_size_rate = 0.3

  print(df)

  # Drop Features and NA
  df.drop(columns=drop_columns, inplace=True)
  df.dropna(inplace=True)

  # Separate X (data), y (target)
  X_data = df.iloc[:, 2:-1]
  y_data = df.iloc[:, -1]

  # Get columns
  columns = df.columns[2:-1]

  # Train Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X_data,
      y_data,
      random_state=0,
      test_size=test_size_rate,
      shuffle=True)
  
  # For binary classification
  objective = "binary:logistic"
  eval_metric = "logloss"
  eval_metric_list = ["error", "logloss"]

  # Evaluation
  eval_metric = "aucpr"
  eval_metric_list.append(eval_metric)
  scoring = 'precision'

  # Build Classification Model with Initial Hyperparams
  classifier = XGBClassifier(
    objective=objective,
    booster="gbtree",
    # eval_metric=eval_metric,
    n_estimators=ne,
    learning_rate=lr,
    max_depth=md,
    subsample=0.8,
    colsample_bytree=1,
    gamma=gm,
    random_state=1,
  )

  # Fit Model
  eval_set = [(X_train, y_train), (X_test, y_test)]
  classifier.fit(
    X_train,
    y_train,
    eval_metric=eval_metric_list,
    eval_set=eval_set,
    verbose=False,
  )

  # Extract predictions
  train_yhat = classifier.predict(X_train)
  test_yhat = classifier.predict(X_test)

  # Set K-Fold Cross Validation levels
  cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=1)

  # Training Results
  train_cross_val_score = cross_val_score(classifier, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

  # Retrieve performance metrics
  training_results = classifier.evals_result()
  validation_0_error = training_results['validation_0'][eval_metric_list[0]]
  validation_1_error = training_results['validation_1'][eval_metric_list[0]]
  validation_0_logloss = training_results['validation_0'][eval_metric_list[1]]
  validation_1_logloss = training_results['validation_1'][eval_metric_list[1]]
  validation_0_auc = training_results['validation_0'][eval_metric_list[2]]
  validation_1_auc = training_results['validation_1'][eval_metric_list[2]]

  # Classification Summary Report
  train_summary_report = classification_report(y_train, train_yhat, output_dict=True, zero_division=False)
  test_summary_report = classification_report(y_test, test_yhat, output_dict=True, zero_division=False)

  # pprint(train_summary_report)
  # pprint(test_summary_report)

  # Feature importance
  importance_features = classifier.feature_importances_

  # Standard deviation
  std_dev_perc = train_cross_val_score.std() * 100
  avg_score_perc = train_cross_val_score.mean() * 100
  print(std_dev_perc)
  print(avg_score_perc)
  print(train_summary_report["1.0"]["precision"])
  print(test_summary_report["1.0"]["precision"])

  # Plots
  # plt.title('Error')
  # plt.plot(validation_0_error)
  # plt.plot(validation_1_error)
  # plt.show()

  plt.title('AUC')
  plt.plot(validation_0_auc)
  plt.plot(validation_1_auc)
  plt.show()

  # plt.title('Feature Importance')
  # plt.bar(columns, importance_features)
  # plt.show()
