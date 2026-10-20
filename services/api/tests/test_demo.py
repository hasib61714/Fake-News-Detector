from collections.abc import Callable

from fastapi.testclient import TestClient

from tests.conftest import REAL_TEXT

URL = "/api/v1/demo/analyze"


def test_demo_endpoint_absent_by_default(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        assert client.post(URL, json={"text": REAL_TEXT}).status_code == 404
        assert URL not in client.get("/openapi.json").json()["paths"]


def test_demo_is_labeled_and_never_touches_the_model(
    make_client: Callable[..., TestClient],
) -> None:
    with make_client(enable_demo_mode=True) as client:
        demo = client.post(URL, json={"text": REAL_TEXT})
        health = client.get("/api/v1/health").json()
        real = client.post("/api/v1/analyze", json={"text": REAL_TEXT})

    assert demo.status_code == 200
    body = demo.json()
    assert body["mode"] == "demo"
    assert body["model"] is None
    assert "NOT produced by the machine-learning model" in body["disclaimer"]
    assert health["demo_mode_enabled"] is True
    # Demo mode does not make the real endpoint pretend a model exists.
    assert health["model"]["status"] == "not_trained"
    assert real.status_code == 503


def test_demo_output_is_independent_of_input(make_client: Callable[..., TestClient]) -> None:
    with make_client(enable_demo_mode=True) as client:
        a = client.post(URL, json={"text": "a" * 60}).json()
        b = client.post(URL, json={"text": "b" * 60}).json()

    assert a == b
