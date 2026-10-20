from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FAKE_TEXT, REAL_TEXT

URL = "/api/v1/analyze"


def test_model_not_trained_returns_clear_503(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        response = client.post(URL, json={"text": REAL_TEXT})

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_not_trained"
    assert "Model not trained yet" in error["message"]


def test_predictions_follow_the_model_not_a_constant(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    with make_client() as client:
        fake = client.post(URL, json={"text": FAKE_TEXT})
        real = client.post(URL, json={"text": REAL_TEXT})

    assert fake.status_code == real.status_code == 200
    assert fake.json()["label"] == "fake"
    assert real.json()["label"] == "real"
    for body in (fake.json(), real.json()):
        assert body["mode"] == "model"
        probs = body["probabilities"]
        assert probs["fake"] + probs["real"] == pytest.approx(1.0)
        assert body["confidence"] == pytest.approx(max(probs.values()))
        assert body["model"]["name"] == "toy-test-model"
        assert body["disclaimer"]


def test_model_trained_after_startup_is_picked_up(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    with make_client() as client:
        assert client.post(URL, json={"text": FAKE_TEXT}).status_code == 503
        write_model()
        assert client.post(URL, json={"text": FAKE_TEXT}).status_code == 200


def test_corrupt_model_file_is_reported_not_swallowed(
    make_client: Callable[..., TestClient], model_path: Path
) -> None:
    model_path.write_bytes(b"not a pickle")
    with make_client() as client:
        response = client.post(URL, json={"text": FAKE_TEXT})
        health = client.get("/api/v1/health").json()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"
    assert health["model"]["status"] == "error"
    assert str(model_path) not in response.text


def test_artifact_with_wrong_classes_is_rejected(
    make_client: Callable[..., TestClient], model_path: Path
) -> None:
    import joblib

    from tests.conftest import fit_toy_pipeline

    joblib.dump(fit_toy_pipeline(labels=("spam", "ham")), model_path)
    with make_client() as client:
        response = client.post(URL, json={"text": FAKE_TEXT})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "too short"},
        {"text": " " * 100},
        {"text": "x" * 20_001},
        {"text": 123},
    ],
)
def test_invalid_requests_get_422_with_typed_error(
    make_client: Callable[..., TestClient], payload: dict[str, object]
) -> None:
    with make_client() as client:
        response = client.post(URL, json=payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["fields"][0]["field"] == "text"


def test_validation_runs_before_model_check(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        assert client.post(URL, json={"text": "short"}).status_code == 422
