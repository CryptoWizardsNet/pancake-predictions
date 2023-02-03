import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from constants import SAVE_HISTORY, RUN_XGB
from func_read import save_history
from func_xgboost import run_xgb

if __name__ == "__main__":

  if SAVE_HISTORY:
    save_history()

  if RUN_XGB:
    # run_xgb("bull_bear_ratio_up")
    run_xgb("bulls_win")


