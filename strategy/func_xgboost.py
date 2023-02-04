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
  md = 3
  lr = 0.01
  gm = 0.03
  test_size_rate = 0.3

  # Set targets and params
  df = pd.read_json("history.json")
  if target_item == "model_ratio":
    df.loc[df["bull_ratio"].shift(-1) > df["bear_ratio"].shift(-1), "TARGET"] = 1
    df.loc[df["bull_ratio"].shift(-1) <= df["bear_ratio"].shift(-1), "TARGET"] = 0
    drop_columns = ["minute", "second", "close_price", "lock_price", "hour"]
    ne = 25
  
  if target_item == "model_direction":
    df.loc[df["close_price"].shift(-1) > df["lock_price"].shift(-1) , "TARGET"] = 1
    df.loc[df["close_price"].shift(-1) <= df["lock_price"].shift(-1) , "TARGET"] = 0
    drop_columns = ["close_price", "lock_price"]
    ne = 20

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

  # Show key metrics
  print("")
  print(target_item)
  print("Std Dev %: ", std_dev_perc)
  print("Train: ", train_summary_report["1.0"]["precision"])
  print("Test: ", test_summary_report["1.0"]["precision"])
  print("")

  # Plots
  # plt.title('Error')
  # plt.plot(validation_0_error)
  # plt.plot(validation_1_error)
  # plt.show()

  # plt.title('AUC')
  # plt.plot(validation_0_auc)
  # plt.plot(validation_1_auc)
  # plt.show()

  # plt.title('Feature Importance')
  # plt.bar(columns, importance_features)
  # plt.show()

  # Save model
  classifier.save_model(f"{target_item}.json")


# Predict XGB payout ratio winnder
def xgb_predict_ratio(df):
  drop_columns = ["minute", "second", "close_price", "lock_price", "hour"]

  # Drop Columns
  df.drop(columns=drop_columns, inplace=True)

  # Prepare data
  X_data = df.iloc[:, 2:]

  # Make predictions
  xbg_classifier = XGBClassifier()
  xbg_classifier.load_model("model_ratio.json")
  preds = xbg_classifier.predict(X_data)
  preds_proba = xbg_classifier.predict_proba(X_data)
  return preds[0], preds_proba.tolist()[0][1]


# Predict XGB payout direction winnder
def xgb_predict_direction(df_2):
  drop_columns = ["close_price", "lock_price"]

  # Drop Columns
  df_2.drop(columns=drop_columns, inplace=True)

  # Prepare data
  X_data = df_2.iloc[:, 2:]

  # Make predictions
  xbg_classifier = XGBClassifier()
  xbg_classifier.load_model("model_direction.json")
  preds = xbg_classifier.predict(X_data)
  preds_proba = xbg_classifier.predict_proba(X_data)
  return preds[0], preds_proba.tolist()[0][1]