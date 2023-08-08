from typing import Any
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import paths
from data_models.infer_request_model import get_inference_request_body_model
from logger import log_error
from serve_utils import (
    ModelResources,
    combine_predictions_response_with_explanations,
    generate_unique_request_id,
    get_model_resources,
    logger,
    transform_req_data_and_make_predictions,
)
from xai.explainer import get_explanations_from_explainer
app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "Server is up and running"}

if __name__ == "__main__":
    uvicorn.run("serve:app",reload=True)