import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib
from skopt import gp_minimize
from config import paths
from logger import get_logger
import os
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config.paths import OPT_HPT_DIR_PATH

np.int = np.int_


HPT_RESULTS_FILE_NAME = "HPT_results.csv"

logger = get_logger(task_name="tune")


def run_hyperparameter_tuning(train_X , train_Y):


    space  = [
              Integer(5, 120, name='n_estimators'),
              Integer(1, 10, name='max_depth'),
              Integer(2,20, name = 'min_samples_split'),
              Integer(1,10, name = 'min_samples_leaf')
             ]
    
    @use_named_args(space)
    def objective(**params):
        rf.set_params(**params)
        return -np.mean(cross_val_score(rf, train_X, train_Y, cv=10, n_jobs=-1,scoring="f1"))


    rf = RandomForestClassifier(max_features="log2")
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=42)

    # print("Best Hyper Parameter : ")
    # print("n_estimators : ",res_gp.x[0])
    # print("max_depth : ",res_gp.x[1])
    best_hyperparameters = {"n_estimators":res_gp.x[0] , "max_depth":res_gp.x[1] , "min_samples_split":res_gp.x[2] , "min_samples_leaf":res_gp.x[3] , "max_features":"log2" }
    
    # Making data hyper paramters directory
    if not os.path.exists(paths.OPT_HPT_DIR_PATH):
        os.makedirs(paths.OPT_HPT_DIR_PATH)
    
    joblib.dump(best_hyperparameters,OPT_HPT_DIR_PATH+"/optimized_hyper_parameters.joblib")
    return best_hyperparameters

    
    




