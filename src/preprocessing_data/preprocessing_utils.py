import joblib
import pandas as pd
import numpy as np
from config import paths
import json

def save_correlated_features(correlated_features):
    try:
        context = {"columns":correlated_features}
        joblib.dump(context,paths.DATA_ARTIFACTS_DIR_PATH+'/correlated_features.joblib')
    except Exception as e:
        raise f"Error while saving correlated features : {e}"
    
def load_correlated_features():
    try:
        context = joblib.dump(paths.DATA_ARTIFACTS_DIR_PATH+'/correlated_features.joblib')
        return context['columns']
        
    except Exception as e:
        return []

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

def save_categorical_imputer(categorical_imputer):
    try:
        joblib.dump(categorical_imputer,paths.DATA_ARTIFACTS_DIR_PATH+'/categorical_imputer.joblib')
    except Exception as e:
        raise f"Error occurs while saving categorical imputor  : {e}"

def save_one_hot_encoder(one_hot_encoder):
    try:
        joblib.dump(one_hot_encoder, paths.DATA_ARTIFACTS_DIR_PATH+'/one_hot_encoder.joblib')
    except Exception as e:
        raise f"Error occurs while saving one hot encoder : {e}"
    
def save_categorical_columns_to_be_considered(categorical_columns_to_be_considered):
    try:
        
        context = {"columns":categorical_columns_to_be_considered}
        joblib.dump(context,paths.DATA_ARTIFACTS_DIR_PATH+'/categorical_context.joblib')
        
    except Exception as e:
        raise f"Error while saving categorical columns context : {e}"
    
def load_categorical_columns_to_be_considered():
    try:
        
        context = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+'/categorical_context.joblib')
        return context['columns']
        
    except Exception as e:
        return []

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

def load_one_hot_encoder():
    try:
        one_hot_encoder = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+"/one_hot_encoder.joblib")
        return one_hot_encoder
    except Exception as e:
        raise f"Error while loading one hot encoder : {e}"
    

def load_categorical_imputor():
    try:
        categorical_imputor = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+"/categorical_imputer.joblib")
        return categorical_imputor
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

def load_numeric_imputor():
    try:
        numeric_imputor = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+"/numeric_imputer.joblib")
        return numeric_imputor
    except Exception as e:
        raise f"Error while loading numeric imputor : {e}"

def load_min_max_scaler():
    try:
        min_max_scaler = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+"/min_max_scaler.joblib")
        return min_max_scaler
    except Exception as e:
        raise f"Error while loading min max scaler : {e}"

def save_numeric_imputor(numeric_imputor):
    try:
        joblib.dump(numeric_imputor, paths.DATA_ARTIFACTS_DIR_PATH+"/numeric_imputer.joblib")
    except Exception as e:
        raise f"Error while saving numeric imputor : {e}"

def sav_min_max_scaler(min_max_scaler ):
    try:
        joblib.dump(min_max_scaler, paths.DATA_ARTIFACTS_DIR_PATH+"/min_max_scaler.joblib")
    except Exception as e:
        raise f"Error while saving min max scaler : {e}"

def save_numeric_columns_to_be_considered(numeric_columns_to_be_considered):
    try:
        context = {"columns":numeric_columns_to_be_considered}
        joblib.dump(context,paths.DATA_ARTIFACTS_DIR_PATH+'/numeric_context.joblib')
    except Exception as e:
        raise f"Error while saving categorical columns context : {e}"

def load_numeric_columns_to_be_considered():
    try:
        context = joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+'/numeric_context.joblib')
        return context['columns']
    except Exception as e:
        return []


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



def perfrom_min_max_scaling(min_max_scaler , numeric_data):
    try:
        min_max_scaler.fit(numeric_data)
        numeric_data = pd.DataFrame(min_max_scaler.transform(numeric_data), columns=numeric_data.columns)
        return min_max_scaler , numeric_data
    except Exception as e:
        raise f"Error occured while performing min max scaling : {e}"


def initiate_processing_pipeline(pipeline , data):
    try:
        pipeline.fit(data)
        return pipeline , pipeline.transform(data)
    except Exception as e:
        raise f"Error occured while intiating pipeline : {e}"

def save_pipeline(pipeline , tag):
    try:
        joblib.dump(pipeline , paths.DATA_ARTIFACTS_DIR_PATH+"/"+tag+"_pipeline.joblib")
    except Exception as e:
        print(e)
        raise f"Error occured while saving pipeline : {e}"


def load_pipeline(tag):
    try:
        return joblib.load(paths.DATA_ARTIFACTS_DIR_PATH+"/"+tag+"_pipeline.joblib")
    except Exception as e:
        raise f"Error occured while saving pipeline : {e}"

# def compile_pipeline(pipeline_categorical , pipeline_numeric , data):
#     pipeline_categorical , transformed_data_categorical = initiate_processing_pipeline(pipeline_categorical , data)
#     pipeline_numeric , transformed_data_numeric = initiate_processing_pipeline(pipeline_numeric , data)
#     transformed_data_categorical.reset_index(drop=True,inplace=True)
#     transformed_data_numeric.reset_index(drop=True,inplace=True)
#     columns = list(transformed_data_categorical.columns) + list(transformed_data_numeric.columns)
#     processed_data = pd.concat([transformed_data_categorical,transformed_data_numeric],axis=1,ignore_index=True)
#     processed_data.columns = columns
#     return pipeline_categorical , pipeline_numeric , processed_data
    