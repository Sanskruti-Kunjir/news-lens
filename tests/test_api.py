"""API tests. The model here is trained on a tiny SYNTHETIC fixture, so these
tests verify the API behaviour only, not real-world model quality."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from ml import prepare_dataset as prep
from ml import train
from tests.test_ml_pipeline import FAST_CANDIDATES, FAST_TFIDF, make_raw


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    df, _ = prep.clean_dataset(make_raw())
    train.run_experiment(df, out_dir=d, tfidf_params=FAST_TFIDF, candidates=FAST_CANDIDATES, make_plots=False)
    return d


@pytest.fixture
def client(artifacts, tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSLENS_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("NEWSLENS_DB_PATH", str(tmp_path / "h.db"))
    with TestClient(app) as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/api/model-info").json()
    assert info["n_categories"] == 3 and "model_version" in info


def test_predict_valid(client):
    r = client.post("/api/predict", json={"headline": "senate election vote", "short_description": "policy"})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_category"] == "POLITICS"
    assert body["score_type"] in ("probability", "decision_score")
    assert 1 <= len(body["candidates"]) <= 3
    assert body["candidates"][0]["category"] == body["predicted_category"]
    assert body["history_id"] is not None
    assert body["explanation"]["terms"]


@pytest.mark.parametrize("payload", [
    {"headline": "   "},
    {"headline": ""},
    {"headline": "ok", "input_mode": "bogus"},
    {"headline": "x" * 501},
    {"headline": "ok", "short_description": "y" * 2001},
])
def test_predict_validation_errors(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_unsupported_mode_returns_400(client):
    r = client.post("/api/predict", json={"headline": "senate vote", "input_mode": "headline"})
    assert r.status_code == 400  # fixture model was trained with input_mode=combined


def test_missing_artifacts_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSLENS_ARTIFACTS_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("NEWSLENS_DB_PATH", str(tmp_path / "h.db"))
    with TestClient(app) as c:
        assert c.get("/api/health").json()["model_loaded"] is False
        assert c.post("/api/predict", json={"headline": "hello"}).status_code == 503
        assert c.get("/api/analytics/metrics").status_code == 503


def test_metrics_artifacts(client):
    m = client.get("/api/analytics/metrics").json()
    assert m["evaluation_split"] == "test"
    assert client.get("/api/analytics/model-comparison").json()["evaluation_split"] == "validation"


def test_history_empty_create_filter_delete(client):
    assert client.get("/api/history").json()["total"] == 0
    a = client.post("/api/predict", json={"headline": "senate election vote"}).json()["history_id"]
    client.post("/api/predict", json={"headline": "team match goal"})
    assert client.get("/api/history").json()["total"] == 2
    assert client.get("/api/history", params={"search": "senate"}).json()["total"] == 1
    assert client.get("/api/history", params={"category": "SPORTS"}).json()["total"] == 1
    assert client.get("/api/history", params={"order": "bad"}).status_code == 422
    assert client.get(f"/api/history/{a}").json()["headline"] == "senate election vote"
    assert client.delete(f"/api/history/{a}").status_code == 200
    assert client.delete(f"/api/history/{a}").status_code == 404
    assert client.delete("/api/history").json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0


def test_history_is_safe_against_sql_injection_strings(client):
    client.post("/api/predict", json={"headline": "senate vote"})
    r = client.get("/api/history", params={"search": "'; DROP TABLE prediction_history; --"})
    assert r.status_code == 200 and r.json()["total"] == 0
    assert client.get("/api/history").json()["total"] == 1
