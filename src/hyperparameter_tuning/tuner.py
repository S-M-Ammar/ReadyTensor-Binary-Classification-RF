import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib
from skopt import gp_minimize
from config import paths
from logger import get_logger
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config.paths import OPT_HPT_DIR_PATH

np.int = np.int_


HPT_RESULTS_FILE_NAME = "HPT_results.csv"

logger = get_logger(task_name="tune")

def logger_callback(res):
    """
    Logger callback for the hyperparameter tuning trials.

    Logs each trial to the logge including:
        - Iteration number
        - Current hyperparameter trial
        - Current trial objective function value
        - Best hyperparameters found so far
        - Best objective function value found so far
    """
    logger.info(f"Iteration: {len(res.x_iters)}")
    logger.info(f"Current trial hyperparameters: {res.x_iters[-1]}")
    logger.info(f"Current trial obj func value: {res.func_vals[-1]}")
    logger.info(f"Best trial hyperparameters: {res.x}")
    logger.info(f"Best objective func value: {res.fun}")



def run_hyperparameter_tuning(train_X , train_Y):


    space  = [
              Integer(5, 120, name='n_estimators'),
              Integer(1, 10, name='max_depth'),
             ]
    
    @use_named_args(space)
    def objective(**params):
        rf.set_params(**params)
        return -np.mean(cross_val_score(rf, train_X, train_Y, cv=5, n_jobs=-1,scoring="neg_mean_absolute_error"))


    rf = RandomForestClassifier()
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

    # print("Best Hyper Parameter : ")
    # print("n_estimators : ",res_gp.x[0])
    # print("max_depth : ",res_gp.x[1])
    best_hyperparameters = {"n_estimators":res_gp.x[0] , "max_depth":res_gp.x[1]}
    joblib.dump(best_hyperparameters,OPT_HPT_DIR_PATH+"/optimized_hyper_parameters.joblib")
    return best_hyperparameters

    
    




