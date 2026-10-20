from collections.abc import Callable

from fastapi.testclient import TestClient


def test_health_without_model(make_client: Callable[..., TestClient]) -> None:
    with make_client() as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == {
        "status": "not_trained",
        "loaded": False,
        "detail": "Model not trained yet.",
        "info": None,
    }
    assert body["demo_mode_enabled"] is False


def test_health_with_model(
    make_client: Callable[..., TestClient], write_model: Callable[[], None]
) -> None:
    write_model()
    with make_client() as client:
        body = client.get("/api/v1/health").json()

    assert body["model"]["status"] == "ready"
    assert body["model"]["loaded"] is True
    info = body["model"]["info"]
    assert info["name"] in {"tfidf-logistic-regression", "tfidf-linear-svm-calibrated"}
    assert info["version"] and info["trained_at"]


def test_cors_allows_configured_origin(make_client: Callable[..., TestClient]) -> None:
    with make_client(cors_origins="http://localhost:3000") as client:
        response = client.options(
            "/api/v1/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        blocked = client.options(
            "/api/v1/analyze",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
        )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in blocked.headers
