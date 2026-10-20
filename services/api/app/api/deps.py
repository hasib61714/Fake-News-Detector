from typing import Annotated, cast

from fastapi import Depends, Request

from app.config import Settings
from app.inference import ModelService


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_model_service(request: Request) -> ModelService:
    return cast(ModelService, request.app.state.model_service)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
