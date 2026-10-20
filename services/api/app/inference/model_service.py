"""Loads the trained, versioned model artifact and runs real inference with it.

The artifact (see ``fakenews_ml.artifacts``) bundles a scikit-learn pipeline, which already
contains the text preprocessing, with metadata: SHA-256, preprocessing version, uncertainty
threshold, evaluation results. ``fakenews_ml.artifacts.load_artifact`` verifies all of that; any
failure leaves the service in the ``error`` state and requests get a 503, never a guess.

SECURITY: joblib uses pickle. Only load artifacts you trained yourself or fully trust;
``MODEL_PATH`` must never be user-controlled.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from fakenews_ml.artifacts import (
    ArtifactError,
    ArtifactNotFoundError,
    LoadedArtifact,
    load_artifact,
)

from app.schemas import Label, ModelHealth, ModelInfo, ModelStatus, Probabilities, Verdict

logger = logging.getLogger(__name__)

_Signature = tuple[float | None, float | None]


class ModelUnavailableError(Exception):
    """Base class: the model cannot produce a prediction right now."""


class ModelNotTrainedError(ModelUnavailableError):
    """No model artifact exists yet; train one (see ml/README.md)."""


class ModelLoadError(ModelUnavailableError):
    """An artifact exists but could not be loaded or failed verification."""


@dataclass(frozen=True)
class Prediction:
    label: Verdict
    top_label: Label
    confidence: float
    probabilities: Probabilities
    uncertainty_threshold: float
    model: ModelInfo


class ModelService:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._artifact: LoadedArtifact | None = None
        self._status: ModelStatus = "not_trained"
        self._detail = "Model not trained yet."
        self._attempted = False
        self._last_signature: _Signature = (None, None)
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------

    def load(self) -> None:
        """(Re)load the artifact. Never raises; inspect ``health()`` afterwards."""
        with self._lock:
            self._load_locked()

    def refresh(self) -> None:
        """Reload if the model or metadata file appeared or changed since the last attempt.

        This lets the API pick up a newly trained model without a restart. A cheap stat() of two
        files; nothing is re-read while they are unchanged.
        """
        if self._attempted and self._signature() == self._last_signature:
            return
        with self._lock:
            if self._attempted and self._signature() == self._last_signature:
                return
            self._load_locked()

    def _signature(self) -> _Signature:
        def mtime(path: Path) -> float | None:
            try:
                return path.stat().st_mtime
            except OSError:
                return None

        return mtime(self._model_path), mtime(self._metadata_path)

    def _load_locked(self) -> None:
        # Taken before reading: a file replaced mid-load triggers one more reload next time.
        self._last_signature = self._signature()
        self._attempted = True
        self._artifact = None

        try:
            artifact = load_artifact(self._model_path, self._metadata_path)
        except ArtifactNotFoundError:
            self._status = "not_trained"
            self._detail = "Model not trained yet."
            logger.warning(
                "No model artifact at %s; train one (see ml/README.md).", self._model_path
            )
            return
        except ArtifactError as exc:
            logger.error("Model artifact at %s failed verification: %s", self._model_path, exc)
            self._status = "error"
            self._detail = "A model file exists but could not be loaded. Retrain the model."
            return
        except Exception:
            logger.exception("Unexpected error loading model artifact at %s", self._model_path)
            self._status = "error"
            self._detail = "A model file exists but could not be loaded. Retrain the model."
            return

        self._artifact = artifact
        self._status = "ready"
        self._detail = "Model loaded."
        logger.info("Loaded model %s (%s)", artifact.name, artifact.version)

    # -- status ----------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._status == "ready"

    def health(self) -> ModelHealth:
        self.refresh()
        artifact = self._artifact
        return ModelHealth(
            status=self._status,
            loaded=self.is_ready,
            detail=self._detail,
            info=_info(artifact) if artifact else None,
        )

    # -- inference -------------------------------------------------------

    def predict(self, text: str) -> Prediction:
        self.refresh()
        artifact = self._artifact
        if self._status == "not_trained":
            raise ModelNotTrainedError(self._detail)
        if self._status == "error" or artifact is None:
            raise ModelLoadError(self._detail)

        pipeline = artifact.pipeline
        probs = pipeline.predict_proba([text])[0]
        by_label = {str(c): float(p) for c, p in zip(pipeline.classes_, probs, strict=True)}
        top_label: Label = "fake" if by_label["fake"] >= by_label["real"] else "real"
        confidence = by_label[top_label]
        verdict: Verdict = "uncertain" if confidence < artifact.threshold else top_label
        return Prediction(
            label=verdict,
            top_label=top_label,
            confidence=confidence,
            probabilities=Probabilities(fake=by_label["fake"], real=by_label["real"]),
            uncertainty_threshold=artifact.threshold,
            model=_info(artifact),
        )


def _info(artifact: LoadedArtifact) -> ModelInfo:
    return ModelInfo(name=artifact.name, version=artifact.version, trained_at=artifact.trained_at)
