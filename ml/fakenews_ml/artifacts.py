"""The model artifact: the fitted pipeline together with the metadata that describes it.

Layout written by training::

    <output-dir>/releases/<model_version>/model.joblib           immutable, one per training run
    <output-dir>/releases/<model_version>/model.metadata.json
    <output-dir>/model.joblib, model.metadata.json                the ACTIVE copy the API loads

The pipeline already contains the preprocessing step, so cleaning, TF-IDF and the classifier
are one object and cannot drift apart. The metadata records the SHA-256 of the model file, the
preprocessing version, library versions, the dataset fingerprint, the cross-validation results,
the uncertainty rule and the held-out test evaluation. ``load_artifact`` verifies all of it.

SECURITY: ``joblib`` files are pickles. Loading one can execute arbitrary code. Only load
artifacts you trained yourself or fully trust; never let users choose the path.
"""

import hashlib
import io
import json
import logging
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from fakenews_ml import __version__
from fakenews_ml.data import LABELS
from fakenews_ml.preprocessing import DESCRIPTION, PREPROCESSING_VERSION

logger = logging.getLogger(__name__)

METADATA_SCHEMA_VERSION = 2
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model.metadata.json"
RELEASES_DIRNAME = "releases"


class ArtifactError(Exception):
    """The artifact is unreadable, inconsistent or incompatible with this code."""


class ArtifactNotFoundError(ArtifactError):
    """There is no model file at the expected path (nothing has been trained yet)."""


@dataclass(frozen=True)
class Artifact:
    model_bytes: bytes
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedArtifact:
    pipeline: Any
    metadata: dict[str, Any]
    name: str
    version: str
    trained_at: str
    threshold: float


def metadata_path_for(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}.metadata.json")


def build_artifact(pipeline: Any, body: dict[str, Any], *, now: datetime | None = None) -> Artifact:
    """Serialize ``pipeline`` and complete ``body`` with the integrity/versioning fields."""
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    model_bytes = buffer.getvalue()
    sha256 = hashlib.sha256(model_bytes).hexdigest()

    metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "model_name": body["model_name"],
        "model_version": f"{now:%Y%m%dT%H%M%SZ}-{sha256[:8]}",
        "trained_at": now.isoformat(),
        "labels": list(LABELS),
        "artifact": {"file": MODEL_FILENAME, "sha256": sha256, "size_bytes": len(model_bytes)},
        "preprocessing": {"version": PREPROCESSING_VERSION, "description": DESCRIPTION},
        "libraries": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "fakenews_ml": __version__,
        },
        **{k: v for k, v in body.items() if k != "model_name"},
    }
    return Artifact(model_bytes=model_bytes, metadata=metadata)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_release(artifact: Artifact, output_dir: Path, *, activate: bool = True) -> Path:
    """Write the immutable release directory and (optionally) make it the active model."""
    metadata_bytes = (json.dumps(artifact.metadata, indent=2, allow_nan=False) + "\n").encode()
    release_dir = output_dir / RELEASES_DIRNAME / str(artifact.metadata["model_version"])
    release_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(release_dir / MODEL_FILENAME, artifact.model_bytes)
    _atomic_write(release_dir / METADATA_FILENAME, metadata_bytes)

    if activate:
        # Metadata first, model last: a loader that catches the two files out of step sees a
        # SHA-256 mismatch (an error it retries), never a silently mismatched pair.
        _atomic_write(output_dir / METADATA_FILENAME, metadata_bytes)
        _atomic_write(output_dir / MODEL_FILENAME, artifact.model_bytes)
    return release_dir


def load_artifact(model_path: Path, metadata_path: Path | None = None) -> LoadedArtifact:
    """Load and verify an artifact. Raises ArtifactNotFoundError or ArtifactError."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path) if metadata_path else metadata_path_for(model_path)

    if not model_path.is_file():
        raise ArtifactNotFoundError(f"No model file at {model_path}.")
    if not metadata_path.is_file():
        raise ArtifactError(f"Model file exists but its metadata is missing: {metadata_path}.")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Metadata is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("Metadata must be a JSON object.")

    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported metadata schema {metadata.get('schema_version')!r}; "
            f"this code reads schema {METADATA_SCHEMA_VERSION}. Retrain the model."
        )

    model_bytes = model_path.read_bytes()
    expected = (metadata.get("artifact") or {}).get("sha256")
    if hashlib.sha256(model_bytes).hexdigest() != expected:
        raise ArtifactError("Model file does not match the SHA-256 recorded in its metadata.")

    recorded = (metadata.get("preprocessing") or {}).get("version")
    if recorded != PREPROCESSING_VERSION:
        raise ArtifactError(
            f"Model was trained with preprocessing version {recorded!r} but this code is "
            f"version {PREPROCESSING_VERSION!r}. Retrain the model."
        )
    if metadata.get("labels") != list(LABELS):
        raise ArtifactError(f"Metadata labels {metadata.get('labels')!r} are not {list(LABELS)}.")

    threshold = (metadata.get("uncertainty") or {}).get("threshold")
    if not isinstance(threshold, float) or not 0.5 <= threshold <= 1.0:
        raise ArtifactError(f"Invalid uncertainty threshold {threshold!r} in metadata.")
    name, version, trained_at = (
        metadata.get("model_name"),
        metadata.get("model_version"),
        metadata.get("trained_at"),
    )
    if not all(isinstance(v, str) and v for v in (name, version, trained_at)):
        raise ArtifactError("Metadata lacks model_name, model_version or trained_at.")

    trained_with = (metadata.get("libraries") or {}).get("scikit_learn")
    if trained_with != sklearn.__version__:
        logger.warning(
            "Model was trained with scikit-learn %s but %s is installed.",
            trained_with,
            sklearn.__version__,
        )

    try:
        pipeline = joblib.load(io.BytesIO(model_bytes))
    except Exception as exc:
        raise ArtifactError(f"Model file could not be deserialized: {exc}") from exc
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
        raise ArtifactError("Artifact is not a classifier with predict_proba and classes_.")
    if {str(c) for c in pipeline.classes_} != set(LABELS):
        raise ArtifactError(f"Model classes {list(pipeline.classes_)} are not {list(LABELS)}.")

    return LoadedArtifact(
        pipeline=pipeline,
        metadata=metadata,
        name=str(name),
        version=str(version),
        trained_at=str(trained_at),
        threshold=float(threshold),
    )
