from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Keep in sync with apps/web/src/lib/validation.ts.
MIN_ARTICLE_CHARS = 50
MAX_ARTICLE_CHARS = 20_000

Label = Literal["fake", "real"]
ModelStatus = Literal["ready", "not_trained", "error"]
ErrorCode = Literal[
    "invalid_request",
    "model_not_trained",
    "model_load_failed",
    "internal_error",
]


class AnalyzeRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=MIN_ARTICLE_CHARS,
            max_length=MAX_ARTICLE_CHARS,
        ),
    ] = Field(description="Full text of the news article to analyze.")


class Probabilities(BaseModel):
    fake: float = Field(ge=0, le=1)
    real: float = Field(ge=0, le=1)


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    trained_at: str | None = None


class AnalyzeResponse(BaseModel):
    mode: Literal["model", "demo"] = Field(
        description="'model' for real inference; 'demo' for the fixed sample payload."
    )
    label: Label
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted label.")
    probabilities: Probabilities
    model: ModelInfo | None = Field(
        description="Model that produced the result. Null in demo mode."
    )
    analyzed_chars: int
    disclaimer: str


class ModelHealth(BaseModel):
    status: ModelStatus
    loaded: bool
    detail: str
    info: ModelInfo | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    model: ModelHealth
    demo_mode_enabled: bool


class FieldError(BaseModel):
    field: str
    message: str


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    fields: list[FieldError] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
