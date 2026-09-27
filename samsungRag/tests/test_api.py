from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["component"] == "retrieval-controller"


def _payload(**overrides):
    base = {
        "session_id": "S001",
        "turn_id": "T001",
        "transcript": "Find workshop venues in Pune",
        "is_final": False,
        "timestamp": 1726800000.0,
        "language": "en",
        "metadata": {},
    }
    base.update(overrides)
    return base


def test_decide_retrieve_contract():
    resp = client.post("/api/v1/retrieval-controller/decide", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "RETRIEVE"
    assert body["schema_version"] == "1.0"
    assert body["session_id"] == "S001"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["state_version"] >= 1
    assert "trigger" in body and "reason_codes" in body


def test_decide_validation_error():
    # Missing session_id -> 422 from FastAPI/Pydantic.
    bad = _payload()
    del bad["session_id"]
    resp = client.post("/api/v1/retrieval-controller/decide", json=bad)
    assert resp.status_code == 422


def test_decide_no_retrieval_ack():
    resp = client.post(
        "/api/v1/retrieval-controller/decide",
        json=_payload(session_id="SX", turn_id="T9", transcript="Okay, thanks"),
    )
    assert resp.status_code == 200
    assert resp.json()["decision"] == "NO_RETRIEVAL"
