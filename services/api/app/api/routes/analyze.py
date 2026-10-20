from fastapi import APIRouter

from app.api.deps import ModelServiceDep
from app.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse

router = APIRouter(tags=["analyze"])

DISCLAIMER = (
    "This is a statistical text classifier, not a fact-check. It can be wrong and "
    "should not be used as the sole basis for judging whether an article is true."
)


@router.post(
    "/analyze",
    summary="Classify an article as fake, real or uncertain with the trained model",
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request body."},
        503: {
            "model": ErrorResponse,
            "description": "Model not trained yet (`model_not_trained`) or failed to load.",
        },
    },
)
def analyze(payload: AnalyzeRequest, model_service: ModelServiceDep) -> AnalyzeResponse:
    prediction = model_service.predict(payload.text)
    return AnalyzeResponse(
        mode="model",
        label=prediction.label,
        top_label=prediction.top_label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        uncertainty_threshold=prediction.uncertainty_threshold,
        model=prediction.model,
        analyzed_chars=len(payload.text),
        disclaimer=DISCLAIMER,
    )
