"""FastAPI endpoint tests using TestClient."""
import json
import pathlib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture: patch data paths so tests don't need real files
# ---------------------------------------------------------------------------

def _make_features_df(n: int = 500) -> pd.DataFrame:
    np.random.seed(99)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.clip(close, 50, 400)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Date": dates,
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": 5_000_000.0,
        "MA_20": close,
        "MA_50": close,
        "RSI_14": 50.0,
        "MACD": 0.0,
        "MACD_signal": 0.0,
        "BB_mid": close,
        "BB_upper": close * 1.02,
        "BB_lower": close * 0.98,
        "ATR_14": 1.0,
        "Support": close * 0.97,
        "Resistance": close * 1.03,
    })


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    """Write a temp CSV and patch the module-level path in backend.main."""
    df = _make_features_df()
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "AAPL_features.csv"
    df.to_csv(csv_path, index=False)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    comparison = {
        "results": [
            {"algo": "Random", "final_portfolio_value": 98000, "total_return_pct": -2.0,
             "sharpe_ratio": -0.5, "max_drawdown": -0.05, "n_steps": 150},
            {"algo": "DQN", "final_portfolio_value": 110000, "total_return_pct": 10.0,
             "sharpe_ratio": 1.2, "max_drawdown": -0.03, "n_steps": 150},
        ],
        "buy_and_hold": {"algo": "Buy&Hold", "final_portfolio_value": 105000,
                         "total_return_pct": 5.0, "sharpe_ratio": 0.8,
                         "max_drawdown": -0.04, "n_steps": 40},
    }
    (results_dir / "comparison.json").write_text(json.dumps(comparison))

    import backend.main as bm
    monkeypatch.setattr(bm, "DATA_PATH", csv_path)
    monkeypatch.setattr(bm, "RESULTS_PATH", results_dir / "comparison.json")
    monkeypatch.setattr(bm, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(bm, "_df", None)  # reset cached dataframe


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_data_loaded_true(self, client):
        r = client.get("/health")
        assert r.json()["data_loaded"] is True


class TestAlgorithmsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/algorithms")
        assert r.status_code == 200

    def test_has_results_key(self, client):
        r = client.get("/api/algorithms")
        assert "results" in r.json()

    def test_results_list(self, client):
        r = client.get("/api/algorithms")
        assert isinstance(r.json()["results"], list)


class TestDataEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/data")
        assert r.status_code == 200

    def test_has_data_key(self, client):
        r = client.get("/api/data")
        assert "data" in r.json()

    def test_ohlcv_fields(self, client):
        r = client.get("/api/data")
        row = r.json()["data"][0]
        for field in ("date", "open", "high", "low", "close", "volume"):
            assert field in row, f"Missing field: {field}"

    def test_prices_positive(self, client):
        r = client.get("/api/data")
        for row in r.json()["data"]:
            assert row["close"] > 0
            assert row["high"] >= row["low"]


class TestSimulateEndpoint:
    def test_returns_200_random(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 100_000})
        assert r.status_code == 200

    def test_response_has_steps(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 100_000})
        body = r.json()
        assert "steps" in body
        assert isinstance(body["steps"], list)

    def test_steps_non_empty(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 100_000})
        assert len(r.json()["steps"]) > 0

    def test_step_schema(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 50_000})
        step = r.json()["steps"][0]
        for field in ("step", "price", "action", "portfolio_value", "capital", "shares"):
            assert field in step, f"Missing field in step: {field}"

    def test_action_values_valid(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 100_000})
        actions = {s["action"] for s in r.json()["steps"]}
        assert actions.issubset({0, 1, 2})

    def test_capital_reflected(self, client):
        r = client.post("/api/simulate", json={"algo": "Random", "capital": 75_000})
        assert r.json()["capital"] == 75_000.0

    def test_invalid_algo_returns_400(self, client):
        r = client.post("/api/simulate", json={"algo": "NOTREAL", "capital": 100_000})
        assert r.status_code == 400
