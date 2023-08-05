import os

# Path to the root directory which contains the src directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path into the mounted volume:
#   set to environment variable MODEL_INPUTS_OUTPUTS_PATH if it exists
#   else: set to default path which would be <path_to_root>/model_inputs_outputs/
MODEL_INPUTS_OUTPUTS = os.environ.get(
    "MODEL_INPUTS_OUTPUTS_PATH", os.path.join(ROOT_DIR, "model_inputs_outputs/")
)

# Path to inputs
INPUT_DIR = os.path.join(MODEL_INPUTS_OUTPUTS, "inputs")
# File path for input schema file
INPUT_SCHEMA_DIR = os.path.join(INPUT_DIR, "schema")
# Path to data directory inside inputs directory
DATA_DIR = os.path.join(INPUT_DIR, "data")
# Path to training directory inside data directory
TRAIN_DIR = os.path.join(DATA_DIR, "training")
# Path to test directory inside data directory
TEST_DIR = os.path.join(DATA_DIR, "testing")

# Path to model directory
MODEL_PATH = os.path.join(MODEL_INPUTS_OUTPUTS, "model")
# Path to artifacts directory inside model directory
MODEL_ARTIFACTS_PATH = os.path.join(MODEL_PATH, "artifacts")
# Path to saved schema in artifacts directory
SAVED_SCHEMA_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "schema")
# Name of the preprocessing pipeline file
PREPROCESSING_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "preprocessing")
# Name of the predictor model file inside artifacts directory
PREDICTOR_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "predictor")
# Name of the explainer file inside artifacts directory
EXPLAINER_DIR_PATH = os.path.join(MODEL_ARTIFACTS_PATH, "explainer")


# Paths inside the source directory
# Path to source directory
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to config directory
CONFIG_DIR = os.path.join(SRC_DIR, "config")
# Path to model config
MODEL_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "model_config.json")

# print("ROOT : ",ROOT_DIR)

# print("MODEL_INPUTS_OUTPUTS : ",MODEL_INPUTS_OUTPUTS)

# print("INPUT_DIR  : ",INPUT_DIR)

# print("INPUT_SCHEMA_DIR : ",INPUT_SCHEMA_DIR)

# print("DATA_DIR : ",DATA_DIR)

# print("TRAIN_DIR : ",TRAIN_DIR)

# print("TEST_DIR : ",TEST_DIR)