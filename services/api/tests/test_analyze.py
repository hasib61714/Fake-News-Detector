import json
from collections.abc import Callable
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tests.conftest import AMBIGUOUS_TEXT, FAKE_TEXT, REAL_TEXT

URL = "/api/v1/analyze"


def post(client: TestClient, text: str) -> dict[str, object]:
    response = client.post(URL, json={"text": text})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


# -- missing / broken model -----------------------------------------------------


def test_model_not_trained_returns_clear_503(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        response = client.post(URL, json={"text": REAL_TEXT})

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "model_not_trained"
    assert "Model not trained yet" in error["message"]
    assert "label" not in response.json()


def test_model_trained_after_startup_is_picked_up(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    with make_client() as client:
        assert client.post(URL, json={"text": FAKE_TEXT}).status_code == 503
        write_model()
        assert client.post(URL, json={"text": FAKE_TEXT}).status_code == 200


def test_replacing_the_model_file_is_picked_up_without_restart(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    with make_client() as client:
        first = client.get("/api/v1/health").json()["model"]["info"]["version"]

        metadata_path = model_path.with_name("model.metadata.json")
        metadata = json.loads(metadata_path.read_text())
        metadata["model_version"] = "20990101T000000Z-retrained"
        metadata_path.write_text(json.dumps(metadata))

        second = client.get("/api/v1/health").json()["model"]["info"]["version"]

    assert first != second == "20990101T000000Z-retrained"


def test_corrupt_model_file_is_reported_not_swallowed(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    model_path.write_bytes(b"not a pickle")
    with make_client() as client:
        response = client.post(URL, json={"text": FAKE_TEXT})
        health = client.get("/api/v1/health").json()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"
    assert health["model"]["status"] == "error"
    assert str(model_path) not in response.text


def test_model_without_metadata_is_refused(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    model_path.with_name("model.metadata.json").unlink()
    with make_client() as client:
        response = client.post(URL, json={"text": FAKE_TEXT})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"


def test_model_not_matching_its_metadata_hash_is_refused(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    other = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression())]).fit(
        ["fake words here", "real words there"] * 3, ["fake", "real"] * 3
    )
    joblib.dump(other, model_path)
    with make_client() as client:
        response = client.post(URL, json={"text": FAKE_TEXT})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"


def test_preprocessing_version_mismatch_is_refused(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    metadata_path = model_path.with_name("model.metadata.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["preprocessing"]["version"] = "0-old"
    metadata_path.write_text(json.dumps(metadata))

    with make_client() as client:
        assert client.post(URL, json={"text": FAKE_TEXT}).status_code == 503
        assert client.get("/api/v1/health").json()["model"]["status"] == "error"


# -- genuine predictions --------------------------------------------------------


def test_predictions_follow_the_model_not_a_constant(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    with make_client() as client:
        fake = post(client, FAKE_TEXT)
        real = post(client, REAL_TEXT)

    assert (fake["label"], real["label"]) == ("fake", "real")
    for body in (fake, real):
        assert body["mode"] == "model"
        probs = body["probabilities"]
        assert isinstance(probs, dict)
        assert probs["fake"] + probs["real"] == pytest.approx(1.0)
        assert body["confidence"] == pytest.approx(max(probs.values()))
        assert body["top_label"] == body["label"]
        assert isinstance(body["model"], dict)
        assert body["model"]["version"]
        assert body["disclaimer"]


def test_response_reports_the_artifacts_uncertainty_threshold(
    make_client: Callable[..., TestClient], write_model: Callable[[], None], model_path: Path
) -> None:
    write_model()
    recorded = json.loads(model_path.with_name("model.metadata.json").read_text())["uncertainty"]
    with make_client() as client:
        body = post(client, FAKE_TEXT)

    assert body["uncertainty_threshold"] == recorded["threshold"]


def test_low_confidence_articles_are_labeled_uncertain(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    with make_client() as client:
        body = post(client, AMBIGUOUS_TEXT)

    assert body["label"] == "uncertain"
    assert body["top_label"] in {"fake", "real"}  # the lean is still reported
    assert isinstance(body["confidence"], float)
    assert isinstance(body["uncertainty_threshold"], float)
    assert body["confidence"] < body["uncertainty_threshold"]


def test_uncertain_exactly_when_confidence_is_below_the_threshold(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    with make_client() as client:
        for text in (FAKE_TEXT, REAL_TEXT, AMBIGUOUS_TEXT, FAKE_TEXT[:60] + " " + REAL_TEXT[:60]):
            body = post(client, text)
            assert isinstance(body["confidence"], float)
            assert isinstance(body["uncertainty_threshold"], float)
            below = body["confidence"] < body["uncertainty_threshold"]
            assert (body["label"] == "uncertain") is below


def test_html_input_is_cleaned_by_the_saved_preprocessing(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    wrapped = f"<div><p>{FAKE_TEXT}</p> <a href='http://example.org/x'>link</a></div>"
    with make_client() as client:
        assert post(client, wrapped)["label"] == "fake"


def test_publisher_boilerplate_is_removed_by_the_saved_preprocessing(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    """The served model applies the same cleaning it was trained with: a dateline, an image credit
    and a tweet embed around an article change nothing about the prediction."""
    write_model()
    noisy = (
        f"WASHINGTON (Reuters) - {FAKE_TEXT} pic.twitter.com/3vD5nFgE36 "
        "Featured image via Mark Wilson/Getty Images"
    )
    with make_client() as client:
        clean_body = post(client, FAKE_TEXT)
        noisy_body = post(client, noisy)

    assert noisy_body["probabilities"] == clean_body["probabilities"]
    assert noisy_body["label"] == clean_body["label"]


# -- invalid input --------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "too short"},
        {"text": " " * 100},
        {"text": "x" * 20_001},
        {"text": 123},
        {"body": "x" * 100},
    ],
)
def test_invalid_requests_get_422_with_typed_error(
    make_client: Callable[..., TestClient],
    write_model: Callable[[], None],
    payload: dict[str, object],
) -> None:
    write_model()
    with make_client() as client:
        response = client.post(URL, json=payload)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["fields"][0]["field"] == "text"


def test_validation_runs_before_the_model_check(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        assert client.post(URL, json={"text": "short"}).status_code == 422


def test_non_json_body_is_a_422_not_a_crash(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        response = client.post(URL, content="plain text", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422
