from fastapi import APIRouter

from app.demo.sample import build_demo_response
from app.schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse

# Mounted only when ENABLE_DEMO_MODE=true (see app.main.create_app).
router = APIRouter(prefix="/demo", tags=["demo"])


@router.post(
    "/analyze",
    summary="DEMO ONLY: fixed sample response, no ML inference",
    responses={422: {"model": ErrorResponse, "description": "Invalid request body."}},
)
def demo_analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    return build_demo_response(analyzed_chars=len(payload.text))
