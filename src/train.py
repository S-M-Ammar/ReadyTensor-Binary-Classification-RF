import argparse
from config import paths
from xai.logger import get_logger, log_error
from data_models.data_validator import validate_data
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
from preprocessing_data.preprocessing_utils import initiate_processing_pipeline
from preprocessing_data.pipeline import CategoricalTransformer , NumericTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path:str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir:str = paths.TRAIN_DIR,
    run_tuning: bool = False,
    ):
    
    try:
        logger.info("Starting training...")
        # load and save schema
        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

        # load model config
        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        set_seeds(seed_value=model_config["seed_value"])

        # load train data
        logger.info("Loading train data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)

        # validate the data
        logger.info("Validating train data...")
        validated_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )

        # split train data into training and validation sets
        logger.info("Performing train/validation split...")
        train_split, val_split = split_train_val(
            validated_data, val_pct=model_config["validation_split"]
        )

        # train_pipeline_categorical = Pipeline([
        #                         ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,True))
        #                     ])
        
        # train_pipeline_numeric = Pipeline([
        #                         ('NumericTransformer', NumericTransformer(data_schema.numeric_features,True))
        #                     ])
        
        # train_pipeline_categorical , train_transformed_data_categorical = initiate_processing_pipeline(train_pipeline_categorical , train_data)
        # train_pipeline_numeric , train_transformed_data_numeric = initiate_processing_pipeline(train_pipeline_numeric , train_data)
        # train_columns = list(train_transformed_data_categorical.columns) + list(train_transformed_data_numeric.columns)
        # train_data = pd.concat([train_transformed_data_categorical,train_transformed_data_numeric],axis=1,ignore_index=True)
        # train_data.columns = train_columns

        # X_train = train_data
        # Y_train = train_split[[data_schema.target]]

        
        test_val_pipeline_categorical = Pipeline([
                                                   ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,False))
                                                ])
        
        test_val_pipeline_numerical = Pipeline([
                                                   ('NumericTransformer', NumericTransformer(data_schema.numeric_features,False))
                                               ])
        
        test_val_pipeline , test_val_transformed_data_categorical = initiate_processing_pipeline(test_val_pipeline_categorical , val_split)
        print(test_val_transformed_data_categorical)
        # test_val_pipeline_numeric , test_val_transformed_data_numeric = initiate_processing_pipeline(test_val_pipeline_numerical , val_split)
        

    except Exception as e:
        logger.error("Error : ",e)

run_training()