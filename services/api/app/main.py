import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import analyze, demo, health
from app.config import Settings, get_settings
from app.errors import register_exception_handlers
from app.inference import ModelService

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model_service.load()
        yield

    app = FastAPI(
        title="Fake News Detector API",
        version=__version__,
        description="Classifies news article text as fake or real using a trained sklearn model",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = ModelService(settings.model_path, settings.metadata_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    register_exception_handlers(app)

    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health.router)
    v1.include_router(analyze.router)
    if settings.enable_demo_mode:
        v1.include_router(demo.router)
    app.include_router(v1)
    return app


app = create_app()
