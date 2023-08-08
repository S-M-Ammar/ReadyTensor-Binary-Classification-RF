"""
This script contains utility functions/classes that are used in serve.py
"""
import uuid
from typing import Any, Dict, Tuple

import pandas as pd
from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from schema.data_schema import load_saved_schema
from utils import read_json_as_dict
from xai.explainer import load_explainer

logger = get_logger(task_name="serve")




def generate_unique_request_id():
    """Generates unique alphanumeric id"""
    return uuid.uuid4().hex[:10]


async def transform_req_data_and_make_predictions():
    pass
   


def create_predictions_response(
    predictions_df: pd.DataFrame, data_schema: Any, request_id: str
) -> Dict[str, Any]:
    """
    Convert the predictions DataFrame to a response dictionary in required format.

    Args:
        transformed_data (pd.DataFrame): The transfomed input data for prediction.
        data_schema (Any): An instance of the BinaryClassificationSchema.
        request_id (str): Unique request id for logging and tracking

    Returns:
        dict: The response data in a dictionary.
    """
    class_names = data_schema.target_classes
    # find predicted class which has the highest probability
    predictions_df["__predicted_class"] = predictions_df[class_names].idxmax(axis=1)
    sample_predictions = []
    for sample in predictions_df.to_dict(orient="records"):
        sample_predictions.append(
            {
                "sampleId": sample[data_schema.id],
                "predictedClass": str(sample["__predicted_class"]),
                "predictedProbabilities": [
                    round(sample[class_names[0]], 5),
                    round(sample[class_names[1]], 5),
                ],
            }
        )
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": pd.Timestamp.now().isoformat(),
        "requestId": request_id,
        "targetClasses": class_names,
        "targetDescription": data_schema.target_description,
        "predictions": sample_predictions,
    }
    return predictions_response


def combine_predictions_response_with_explanations(
    predictions_response: dict, explanations: dict
) -> dict:
    """
    Combine the predictions response with explanations.

    Inserts explanations for each sample into the respective prediction dictionary
    for the sample.

    Args:
        predictions_response (dict): The response data in a dictionary.
        explanations (dict): The explanations for the predictions.
    """
    for pred, exp in zip(
        predictions_response["predictions"], explanations["explanations"]
    ):
        pred["explanation"] = exp
    predictions_response["explanationMethod"] = explanations["explanation_method"]
    return predictions_response