from fastapi import APIRouter

from app import __version__
from app.api.deps import ModelServiceDep, SettingsDep
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service and model status")
def health(settings: SettingsDep, model_service: ModelServiceDep) -> HealthResponse:
    """Always 200 while the service is up; check ``model.status`` for model readiness."""
    return HealthResponse(
        status="ok",
        service="fake-news-detector-api",
        version=__version__,
        environment=settings.app_env,
        model=model_service.health(),
        demo_mode_enabled=settings.enable_demo_mode,
    )
