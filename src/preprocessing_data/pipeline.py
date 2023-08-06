import pandas as pd
import math
import numpy as np
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from config import paths
from preprocessing_data.preprocessing_utils import *


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, is_training=False , missing_value_threshold_percent=30):
        self.categorical_columns =  categorical_columns
        self.categorical_imputator = None
        self.total_records = None
        self.is_training = is_training
        self.missing_value_threshold_percent = missing_value_threshold_percent
        self.categorical_columns_to_considered = []
        
    def fit(self, X, y=None):
        pass

    def transform(self, data):

        if(len(self.categorical_columns)==0):
            return data
        
        self.total_records = len(data)
        categorical_data = data[self.categorical_columns]
        categorical_data = cast_to_object_categorical_columns(categorical_data)
        self.categorical_columns_to_considered = get_categorical_columns_with_missing_value_threshold(categorical_data,self.categorical_columns,self.total_records,self.missing_value_threshold_percent)
        
        if(len(self.categorical_columns_to_considered) >= 1):
        
            categorical_data = categorical_data[self.categorical_columns_to_considered]
            categorical_imputator = CategoricalImputer(imputation_method='frequent')
            categorical_imputator , transformed_categorical_data = perform_categorical_imputation(categorical_imputator , categorical_data)
            # save_categorical_imputer(categorical_imputator)
            transformed_categorical_data = cast_to_object_categorical_columns(transformed_categorical_data)
            
            one_hot_encoder = OneHotEncoder(drop_last=True , variables=self.categorical_columns_to_considered)
            one_hot_encoder , transformed_categorical_data = perform_one_hot_encoder(one_hot_encoder , transformed_categorical_data)
            # save_one_hot_encoder(one_hot_encoder)
            return transformed_categorical_data
            
        return categorical_data
    



class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns , is_training=False , missing_value_threshold_percent=30):
        self.numeric_columns =  numeric_columns
        self.numeric_imputator = None
        self.total_records = None
        self.is_training = is_training
        self.missing_value_threshold_percent = missing_value_threshold_percent
        self.numeric_columns_to_considered = []

    def fit(self, X, y=None):
        pass

    def transform(self, data):

        if(len(self.numeric_columns)==0):
            return data
        
        self.total_records = len(data)
        numeric_data = data[self.numeric_columns]
        numeric_data = cast_to_object_numeric_columns(numeric_data)
        self.numeric_columns_to_considered = get_numeric_columns_with_missing_value_threshold(numeric_data,self.numeric_columns,self.total_records,self.missing_value_threshold_percent)

        if(len(self.numeric_columns_to_considered) >= 1):
            numeric_data = numeric_data[self.numeric_columns_to_considered]
            numeric_imputator = MeanMedianImputer(imputation_method='median')
            numeric_imputator , transformed_numeric_data = perform_numeric_imputation(numeric_imputator , numeric_data)
            # save_numeric_imputor()

            min_max_scaler = MinMaxScaler()
            min_max_scaler , transformed_numeric_data = perfrom_min_max_scaling(min_max_scaler , transformed_numeric_data)
            # sav_min_max_scaler
            
            return transformed_numeric_data

        return numeric_data