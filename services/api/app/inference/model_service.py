"""Loads the trained scikit-learn pipeline and runs real inference with it.

The artifact is produced by ``ml/`` (see ml/README.md): a joblib-serialized
``sklearn.pipeline.Pipeline`` that maps raw text to class probabilities for the
labels ``"fake"`` and ``"real"``, plus an optional JSON metadata sidecar.

SECURITY: joblib uses pickle. Only load artifacts you trained yourself or fully
trust; ``MODEL_PATH`` must never be user-controlled.
"""

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from app.schemas import Label, ModelHealth, ModelInfo, ModelStatus, Probabilities

logger = logging.getLogger(__name__)

EXPECTED_LABELS = frozenset({"fake", "real"})


class ModelUnavailableError(Exception):
    """Base class: the model cannot produce a prediction right now."""


class ModelNotTrainedError(ModelUnavailableError):
    """No model artifact exists yet; run ``make train``."""


class ModelLoadError(ModelUnavailableError):
    """An artifact exists but could not be loaded or failed validation."""


@dataclass(frozen=True)
class Prediction:
    label: Label
    confidence: float
    probabilities: Probabilities
    model: ModelInfo | None


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._pipeline: Any = None
        self._info: ModelInfo | None = None
        self._status: ModelStatus = "not_trained"
        self._detail = "Model not trained yet."
        self._last_attempt_mtime: float | None = None
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """Try to (re)load the artifact. Never raises; inspect ``health()`` afterwards."""
        with self._lock:
            self._load_locked()

    def refresh(self) -> None:
        """Retry loading if the model is not ready and the artifact is new or changed.

        This lets the API pick up a freshly trained model without a restart.
        """
        if self._status == "ready":
            return
        with self._lock:
            if self.is_ready:
                return
            mtime = self._artifact_mtime()
            if mtime != self._last_attempt_mtime:
                self._load_locked()

    def _artifact_mtime(self) -> float | None:
        try:
            return self._model_path.stat().st_mtime
        except OSError:
            return None

    def _load_locked(self) -> None:
        self._last_attempt_mtime = self._artifact_mtime()
        self._pipeline = None
        self._info = None

        if self._last_attempt_mtime is None:
            self._status = "not_trained"
            self._detail = "Model not trained yet."
            logger.warning("No model artifact at %s; run `make train`.", self._model_path)
            return

        try:
            pipeline = joblib.load(self._model_path)
            self._validate(pipeline)
            self._info = self._read_info()
        except Exception:
            logger.exception("Failed to load model artifact at %s", self._model_path)
            self._status = "error"
            self._detail = "A model file exists but could not be loaded. Retrain the model."
            return

        self._pipeline = pipeline
        self._status = "ready"
        self._detail = "Model loaded."
        logger.info("Loaded model artifact from %s", self._model_path)

    @staticmethod
    def _validate(pipeline: Any) -> None:
        if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
            raise ValueError("Artifact is not a classifier with predict_proba/classes_.")
        classes = {str(c) for c in pipeline.classes_}
        if classes != EXPECTED_LABELS:
            raise ValueError(f"Unexpected classes {sorted(classes)}; expected fake/real.")

    def _read_info(self) -> ModelInfo:
        metadata: dict[str, Any] = {}
        if self._metadata_path.exists():
            loaded = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        return ModelInfo(
            name=str(metadata.get("model_name", "unnamed-model")),
            version=_optional_str(metadata.get("model_version")),
            trained_at=_optional_str(metadata.get("trained_at")),
        )

    # -- status ----------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._status == "ready"

    def health(self) -> ModelHealth:
        self.refresh()
        return ModelHealth(
            status=self._status,
            loaded=self.is_ready,
            detail=self._detail,
            info=self._info,
        )

    # -- inference -------------------------------------------------------

    def predict(self, text: str) -> Prediction:
        self.refresh()
        if self._status == "not_trained":
            raise ModelNotTrainedError(self._detail)
        if self._status == "error" or self._pipeline is None:
            raise ModelLoadError(self._detail)

        pipeline, info = self._pipeline, self._info
        probs = pipeline.predict_proba([text])[0]
        by_label = {str(c): float(p) for c, p in zip(pipeline.classes_, probs, strict=True)}
        label: Label = "fake" if by_label["fake"] >= by_label["real"] else "real"
        return Prediction(
            label=label,
            confidence=by_label[label],
            probabilities=Probabilities(fake=by_label["fake"], real=by_label["real"]),
            model=info,
        )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
