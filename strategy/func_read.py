from func_xgboost import xgb_predict_ratio, xgb_predict_direction
from constants import CONTRACT, NUM_RECORDS_HISTORY
from datetime import datetime
from web3 import Web3
import pandas as pd
import time
import json


# Get current epoch
def get_current_epoch():
  return CONTRACT.functions.currentEpoch().call()


# Save historical trade information to use in training
def save_history():
  current_epoch = get_current_epoch()
  check_epoch = current_epoch - NUM_RECORDS_HISTORY

  # Initialize
  records = []
  counts = 0

  # Store record for each epoch
  while check_epoch < current_epoch:
    try:
      current_rounds_list = CONTRACT.functions.rounds(check_epoch).call()
    except Exception as e:
       print("An error occured at: ", check_epoch)
       print(e)
       exit(1)
    
    lock_timestamp = current_rounds_list[2]
    lock_price = current_rounds_list[4]
    close_price = current_rounds_list[5]
    lock_timestamp = current_rounds_list[2]
    total_amount = current_rounds_list[8]
    bull_amount = current_rounds_list[9]
    bear_amount = current_rounds_list[10]

    # Sleep
    time.sleep(0.2)

    # Convert datetime
    date_log = datetime.fromtimestamp(lock_timestamp)

    # Calculate Ratio
    total_amount_normal = round(float(Web3.fromWei(total_amount, "ether")), 5)
    bull_amount_normal = round(float(Web3.fromWei(bull_amount, "ether")), 5)
    bear_amount_normal = round(float(Web3.fromWei(bear_amount, "ether")), 5)

    # Ratios
    if bull_amount_normal != 0 and bear_amount_normal != 0:
        bull_ratio = round(bull_amount_normal / bear_amount_normal, 2) + 1
        bear_ratio = round(bear_amount_normal / bull_amount_normal, 2) + 1

        # Format numbers
        bull_ratio = float(f'{bull_ratio:.{3}g}')
        bear_ratio = float(f'{bear_ratio:.{3}g}')
    else:
        bull_ratio = 0
        bear_ratio = 0

    # Construct item
    item_dict = {
       "epoch": check_epoch,
       "datetime": date_log.strftime('%Y-%m-%d %H:%M:%S'),
       "hour": date_log.hour,
       "minute": date_log.minute,
       "second": date_log.second,
       "lock_price": lock_price,
       "close_price": close_price,
       "total_amount": total_amount_normal,
       "bull_amount": bull_amount_normal,
       "bear_amount": bear_amount_normal,
       "bull_ratio": bull_ratio,
       "bear_ratio": bear_ratio,
    }

    # Add to records
    records.append(item_dict)
    with open('history.json', 'w', encoding='utf-8') as f:
      json.dump(records, f, ensure_ascii=False, indent=4)

    # Increment by 1
    check_epoch += 1
    counts += 1
    print(counts)
  

# Make predictions
def make_predictions():
  current_epoch = get_current_epoch()
  stats_epoch = current_epoch - 1
  try:
    current_rounds_list = CONTRACT.functions.rounds(stats_epoch).call()
  except Exception as e:
      print("An error occured at: ", stats_epoch)
      print(e)
      exit(1)
  
  lock_timestamp = current_rounds_list[2]
  lock_price = current_rounds_list[4]
  close_price = current_rounds_list[5]
  lock_timestamp = current_rounds_list[2]
  total_amount = current_rounds_list[8]
  bull_amount = current_rounds_list[9]
  bear_amount = current_rounds_list[10]

  # Convert datetime
  date_log = datetime.fromtimestamp(lock_timestamp)

  # Calculate Ratio
  total_amount_normal = round(float(Web3.fromWei(total_amount, "ether")), 5)
  bull_amount_normal = round(float(Web3.fromWei(bull_amount, "ether")), 5)
  bear_amount_normal = round(float(Web3.fromWei(bear_amount, "ether")), 5)

  # Ratios
  if bull_amount_normal != 0 and bear_amount_normal != 0:
      bull_ratio = round(bull_amount_normal / bear_amount_normal, 2) + 1
      bear_ratio = round(bear_amount_normal / bull_amount_normal, 2) + 1

      # Format numbers
      bull_ratio = float(f'{bull_ratio:.{3}g}')
      bear_ratio = float(f'{bear_ratio:.{3}g}')
  else:
      bull_ratio = 0
      bear_ratio = 0

  # Construct item
  item_dict = {
    "epoch": stats_epoch,
    "datetime": date_log.strftime('%Y-%m-%d %H:%M:%S'),
    "hour": date_log.hour,
    "minute": date_log.minute,
    "second": date_log.second,
    "lock_price": lock_price,
    "close_price": close_price,
    "total_amount": total_amount_normal,
    "bull_amount": bull_amount_normal,
    "bear_amount": bear_amount_normal,
    "bull_ratio": bull_ratio,
    "bear_ratio": bear_ratio,
  }

  # Construct dataframe
  df = pd.DataFrame([item_dict])
  df_2 = df.copy()

  # Get preds
  ratio_bull_pred, ratio_bull_pred_proba = xgb_predict_ratio(df)
  direction_bear_pred, direction_bear_pred_proba = xgb_predict_direction(df_2)
  print(ratio_bull_pred, ratio_bull_pred_proba)
  print(direction_bear_pred, direction_bear_pred_proba)

  # Define trade
  if ratio_bull_pred == 1:
    pass
    # send short bear (as payout for bears will likely be higher) tx
