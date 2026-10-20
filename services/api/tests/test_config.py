from pathlib import Path

from app.config import Settings


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = Settings(_env_file=None, cors_origins=" http://a.test , http://b.test,, ")  # type: ignore[call-arg]
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_metadata_path_is_next_to_model() -> None:
    settings = Settings(_env_file=None, model_path=Path("/x/y/model.joblib"))  # type: ignore[call-arg]
    assert settings.metadata_path == Path("/x/y/model.metadata.json")
