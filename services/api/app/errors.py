from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import ModelLoadError, ModelNotTrainedError
from app.schemas import ErrorBody, ErrorCode, ErrorResponse, FieldError

MODEL_NOT_TRAINED_MESSAGE = (
    "Model not trained yet. Train the classifier (see ml/README.md) and try again."
)
MODEL_LOAD_FAILED_MESSAGE = (
    "The trained model could not be loaded. Retrain it (see ml/README.md) and try again."
)


def error_response(
    status_code: int,
    code: ErrorCode,
    message: str,
    fields: list[FieldError] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, fields=fields or []))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModelNotTrainedError)
    async def _not_trained(_: Request, __: ModelNotTrainedError) -> JSONResponse:
        return error_response(503, "model_not_trained", MODEL_NOT_TRAINED_MESSAGE)

    @app.exception_handler(ModelLoadError)
    async def _load_failed(_: Request, __: ModelLoadError) -> JSONResponse:
        return error_response(503, "model_load_failed", MODEL_LOAD_FAILED_MESSAGE)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            FieldError(
                field=".".join(str(part) for part in err["loc"] if part != "body"),
                message=str(err["msg"]),
            )
            for err in exc.errors()
        ]
        return error_response(422, "invalid_request", "The request was invalid.", fields)
