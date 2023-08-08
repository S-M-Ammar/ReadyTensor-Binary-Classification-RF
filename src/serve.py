from typing import Any
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import paths
from data_models.infer_request_model import get_inference_request_body_model
from logger import log_error
from xai.explainer import get_explanations_from_explainer
from serve_utils import logger
from schema.data_schema import load_saved_schema


def create_app():

    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        """GET endpoint that returns a message indicating the service is running.

        Returns:
            dict: A dictionary with a "message" key and "Pong!" value.
        """
        logger.info("Received ping request. Service is healthy...")
        return {"message": "Pong!"}

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Any, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle validation errors for FastAPI requests.

        Args:
            request (Any): The FastAPI request instance.
            exc (RequestValidationError): The RequestValidationError instance.
        Returns:
            JSONResponse: A JSON response with the error message and a 400 status code.
        """
        err_msg = "Validation error with request data."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc), "predictions": None},
        )

    InferenceRequestBodyModel = get_inference_request_body_model(load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH))
