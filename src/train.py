import argparse
from config import paths
from logger import get_logger, log_error
from data_models.data_validator import validate_data
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
from preprocessing_data.preprocessing_utils import initiate_processing_pipeline , compile_pipeline ,save_pipeline
from preprocessing_data.pipeline import CategoricalTransformer , NumericTransformer
from prediction.predictor_model import evaluate_predictor_model,save_predictor_model,train_predictor_model
from xai.explainer import fit_and_save_explainer


from sklearn.pipeline import Pipeline
import pandas as pd

logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path:str = paths.MODEL_CONFIG_FILE_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    train_dir:str = paths.TRAIN_DIR,
    explainer_config_file_path: str = paths.EXPLAINER_CONFIG_FILE_PATH,
    explainer_dir_path: str = paths.EXPLAINER_DIR_PATH,
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

        train_pipeline_categorical = Pipeline([
                                              ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,True))
                                             ])
        
        train_pipeline_numeric = Pipeline([
                                            ('NumericTransformer', NumericTransformer(data_schema.numeric_features,True))
                                          ])
        
        test_val_pipeline_categorical = Pipeline([
                                                   ('CategoricalTransformer', CategoricalTransformer(data_schema.categorical_features,False))
                                                ])
        
        test_val_pipeline_numeric = Pipeline([
                                                   ('NumericTransformer', NumericTransformer(data_schema.numeric_features,False))
                                               ])
        train_pipeline_categorical , train_pipeline_numeric , processed_train_data = compile_pipeline(train_pipeline_categorical , train_pipeline_numeric , train_split)
        test_val_pipeline_categorical , test_val_pipeline_numeric , processed_test_val_data = compile_pipeline(test_val_pipeline_categorical , test_val_pipeline_numeric , val_split)

        save_pipeline(train_pipeline_categorical , "train_categorical")
        save_pipeline(train_pipeline_numeric , "train_numeric")
        save_pipeline(test_val_pipeline_categorical,"test_val_categorical")
        save_pipeline(test_val_pipeline_numeric , "test_val_numeric")
        

        X_train = processed_train_data
        Y_train = train_split[[data_schema.target]]
        X_val = processed_test_val_data
        Y_val = val_split[[data_schema.target]]

        logger.info("Training classifier...")
        default_hyperparameters = read_json_as_dict(
            default_hyperparameters_file_path
        )
        predictor = train_predictor_model(
            X_train, Y_train, default_hyperparameters
        )

        logger.info("Saving classifier...")
        save_predictor_model(predictor, predictor_dir_path)

        # calculate and print validation accuracy
        logger.info("Calculating accuracy on validation data...")
        val_accuracy = evaluate_predictor_model(
            predictor, X_val, Y_val
        )
        logger.info(f"Validation data accuracy: {val_accuracy}")

        logger.info("Fitting and saving explainer...")
        _ = fit_and_save_explainer(
            X_train, explainer_config_file_path, explainer_dir_path
        )
        
        logger.info("Training completed successfully")
   
        
        # test_val_pipeline , test_val_transformed_data_categorical = initiate_processing_pipeline(test_val_pipeline_categorical , val_split)
        # print(test_val_transformed_data_categorical)
        # test_val_pipeline_numeric , test_val_transformed_data_numeric = initiate_processing_pipeline(test_val_pipeline_numerical , val_split)
        

    except Exception as exc:
       
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc

run_training()