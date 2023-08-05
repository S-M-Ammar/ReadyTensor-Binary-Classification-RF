import argparse
from config import paths
from xai.logger import get_logger, log_error
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, read_json_as_dict, set_seeds, split_train_val
logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path:str = paths.MODEL_ARTIFACTS_PATH,
    run_tuning: bool = False,
    ):
    
    try:
        print("Script run successfully")
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

    except Exception as e:
        logger.error("Error : ",e)

run_training()