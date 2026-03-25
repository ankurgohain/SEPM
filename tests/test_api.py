# Run with:  pytest tests/test_api.py -v

from __future__ import annotations
import json
import numpy as np 
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.model_registry import ModelRegistry, get_registry
from src.api.schemas import InterventionType, DropoutTier

API_KEY = "dev-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

GOOD_SESSION = {
    'quiz_score': 72.5,
    "engagement_rate": 0.75,
    "hint_count": 2,
    "session_duration": 45.0,
    "correct_attempts": 12,
    "incorrect_attempts": 4,
    "module_id": 0,
}
 
STRUGGLING_SESSION = {
    "quiz_score": 30.0,
    "engagement_rate": 0.15,
    "hint_count": 9,
    "session_duration": 8.0,
    "correct_attempts": 1,
    "incorrect_attempts": 18,
    "module_id": 2,
}
 

class StubRegistry(ModelRegistry):
    def __init__(self):
        super().__init__()
        self._version = "stub_test"
        self._loaded_at = __import__("time").time()
        self._model = object()
        from sklearn.preprocessing import MinMaxScaler
        dummy = np.array([[0,0,0,1,0,0],[100,1,10,240,50,50,]], dtype = np.float32)
        self._scaler = MinMaxScaler().fit(dummy)

    def predict(self, num_sequence, cat_sequence, return_attention = False):
        B= num_sequence.shape[0]
        results = []
        for i in range(B):
            nonzero = num_sequence[i][num_sequence[i, :, 0] > 0]
            raw_score = float(nonzero[-1, 0]) if len(nonzero) else 0.5

            perf = float(np.clip(raw_score*100, 0,  100))
            mastery = float(np.clip(raw_score*0.9, 0,1))
            dropout = float(np.clip(1 - raw_score, 0, 1))
            tier = "high" if dropout >= 0.65 else "medium" if dropout >= 0.40 else "low"
            intervention = self._recommend(mastery, dropout, perf)

            results.append({
                "performance_score": perf,
                "mastery_probability": mastery,
                "dropout_risk": dropout,
                "dropout_tier": tier,
                "intervention": intervention,
                "model_version": self._version
            })
        return results
    
@pytest.fixtures(scope ="module")
def client():
    stub = StubRegistry()

    def override_registry():
        return stub
    
    app.dependency_overrides[get_registry] = override_registry
    with TestClient(app, raise_server_exceptions = False) as c:
        yield c
    app.dependency_overrides.clear()


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body['status'] == 'ok'
        assert body['model_loaded'] is True
        assert 'model_version' in body
        assert 'uptime_seconds' in body

class TestAuth:
    def test_missing_token(self, client):
        r = client.post('/predict/learner', json={'learner_id': 'u1', 'sessions': [GOOD_SESSION]})
        assert r.status_code == 401

    def test_wrong_toekn(self, client):
        r = client.post('/predict/leaner', json = {'learner_id': 'u1', 'sessions':[GOOD_SESSION]}, headers = {'Authorization': "Bearer wrong-ley"},)
        assert r.status_code == 401
    
    def test_valid_token(self, client):
        r = client.post("/predict/learner", json={"learner_id": "u1", "sessions": [GOOD_SESSION]}, headers = AUTH,)
        assert r.status_code == 200