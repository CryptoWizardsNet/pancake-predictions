from constants import SAVE_HISTORY, RUN_MODEL_1, RUN_MODEL_2
from func_read import save_history
from func_xgboost import run_xgb

if __name__ == "__main__":

  if SAVE_HISTORY:
    save_history()

  if RUN_MODEL_1:
    run_xgb("model_ratio")

  if RUN_MODEL_2:
    run_xgb("model_direction")
