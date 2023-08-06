from feature_engine.imputation import CategoricalImputer
import joblib
import pandas as pd
import numpy as np
from config import paths


def get_categorical_columns_with_missing_value_threshold(categorical_data,categorical_columns,total_records,threshold):
    try:
        columns_to_be_considered = []
        for categorical_column in categorical_columns:
            number_of_missing_values = categorical_data[categorical_column].isna().sum()
            missing_value_percentage = round((number_of_missing_values/total_records)*100,2)
            if(missing_value_percentage <= threshold):
                columns_to_be_considered.append(categorical_column)

        return columns_to_be_considered
    except Exception as e:
        raise f"Error occurs while calculating categorical missing values : {e}"

def save_categorical_imputer(categorical_imputer,path):
    try:
        joblib.dump(categorical_imputer, path+'/categorical_imputer.joblib')
    except Exception as e:
        raise f"Error occurs while saving categorical imputor  : {e}"

def save_one_hot_encoder(one_hot_encoder,path):
    try:
        joblib.dump(one_hot_encoder, path+'/one_hot_encoder.joblib')
    except Exception as e:
        raise f"Error occurs while saving one hot encoder : {e}"
    
def save_categorical_columns_to_be_considered(categorical_columns_to_be_considered,path):
    try:
        context = {"columns":categorical_columns_to_be_considered}
        joblib.dump(context,path+'/categorical_context.joblib')
    except Exception as e:
        raise f"Error while saving categorical columns context : {e}"

def cast_to_object_categorical_columns(categorical_data):
    try:
        for categorical_column in categorical_data.columns:
            try:
                categorical_data[categorical_column] = categorical_data[categorical_column].astype(np.int32)
            except:
                pass
            categorical_data[categorical_column] = categorical_data[categorical_column].astype(np.object_)
    
        return categorical_data
    
    except Exception as e:
        raise f"Error occurs while casting categorical columns : {e}"


def perform_categorical_imputation(categorical_imputer , categorical_data):
    try:
        categorical_imputer.fit(categorical_data)
        return categorical_imputer , categorical_imputer.transform(categorical_data)
    except Exception as e:
        raise f"Error occurs while performing categorical imputation : {e}"
    
def perform_one_hot_encoder(one_hot_encoder , categorical_data):
    try:
        one_hot_encoder.fit(categorical_data)
        return one_hot_encoder , one_hot_encoder.transform(categorical_data)
    except Exception as e:
        raise f"Error occurs while performing one hot encoding : {e}"

def load_one_hot_encoder(path):
    try:
        one_hot_encoder = joblib.load(path+"/one_hot_encoder.joblib")
        return one_hot_encoder
    except Exception as e:
        raise f"Error while loading one hot encoder : {e}"
    

def load_categorical_imputor(path):
    try:
        categorical_imputor = joblib.load(path+"/categorical_imputer.joblib")
        return load_categorical_imputor
    except Exception as e:
        raise f"Error while loading one hot encoder : {e}"



def get_numeric_columns_with_missing_value_threshold(numeric_data,numeric_columns,total_records,threshold):
    try:
        columns_to_be_considered = []
        for numeric_column in numeric_columns:
            number_of_missing_values = numeric_data[numeric_column].isna().sum()
            missing_value_percentage = round((number_of_missing_values/total_records)*100,2)
            if(missing_value_percentage <= threshold):
                columns_to_be_considered.append(numeric_column)

        return columns_to_be_considered
    except Exception as e:
        raise f"Error occurs while calculating numeric missing values : {e}"

def load_numeric_imputor(path):
    try:
        numeric_imputor = joblib.load(path+"/numeric_imputer.joblib")
        return load_categorical_imputor
    except Exception as e:
        raise f"Error while loading one hot encoder : {e}"


def cast_to_object_numeric_columns(numeric_data):
    try:
        for numeric_column in numeric_data.columns:

            numeric_data[numeric_column] = numeric_data[numeric_column].astype(np.float32)
        
        return numeric_data

    
    except Exception as e:
        raise f"Error occurs while casting numeric columns : {e}"


def perform_numeric_imputation(numeric_imputer , numeric_data):
    try:
        numeric_imputer.fit(numeric_data)
        return numeric_imputer , numeric_imputer.transform(numeric_data)
    except Exception as e:
        raise f"Error occurs while performing numeric imputation : {e}"



def perfrom_min_max_scaling():
    pass

