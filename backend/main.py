"""FastAPI backend for RiskAssessRL trading dashboard."""
from __future__ import annotations
import asyncio
import json
import pathlib
import random as _random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
RESULTS_PATH = ROOT / "results" / "comparison.json"
MODEL_DIR = ROOT / "models"

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]

# For backwards-compat with test patching
DATA_PATH = PROC_DIR / "AAPL_features.csv"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="RiskAssessRL API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_df: pd.DataFrame | None = None          # backwards-compat (AAPL default)
_dfs: dict[str, pd.DataFrame] = {}


def get_df(ticker: str = "AAPL") -> pd.DataFrame:
    global _df, _dfs
    ticker = ticker.upper()

    # Legacy path used by tests (monkeypatched via DATA_PATH)
    if ticker == "AAPL" and _df is not None:
        return _df

    if ticker not in _dfs:
        # Respect monkeypatched DATA_PATH for AAPL in tests
        if ticker == "AAPL":
            path = DATA_PATH
        else:
            path = PROC_DIR / f"{ticker}_features.csv"

        if not path.exists():
            raise HTTPException(503, f"Data for {ticker} not found. Run the pipeline first.")
        loaded = pd.read_csv(path, parse_dates=["Date"])
        _dfs[ticker] = loaded
        if ticker == "AAPL":
            _df = loaded  # keep legacy ref in sync
    return _dfs[ticker]


def _test_slice(df: pd.DataFrame) -> pd.DataFrame:
    split = int(0.8 * len(df))
    return df.iloc[split:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(algo: str, obs_dim: int, n_actions: int):
    from src.agents import (RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent)

    algo_up = algo.upper()
    if algo_up == "RANDOM":
        class _Stub:
            class action_space:
                n = n_actions
                @staticmethod
                def sample():
                    import random
                    return random.randrange(n_actions)
        return RandomAgent(_Stub())

    model_map = {
        "DQN": (DQNAgent, "dqn.pth"),
        "DDQN": (DoubleDQNAgent, "ddqn.pth"),
        "A2C": (A2CAgent, "a2c.pth"),
        "PPO": (PPOAgent, "ppo.pth"),
    }
    if algo_up not in model_map:
        raise HTTPException(400, f"Unknown algorithm: {algo}")

    cls, fname = model_map[algo_up]
    agent = cls(obs_dim, n_actions)
    p = MODEL_DIR / fname
    if p.exists():
        agent.load(str(p))
    return agent


def _compute_extra_metrics(steps: list[dict], initial_capital: float) -> dict:
    """Compute win_rate and avg_hold_time from the step log.

    Tracks completed buy→sell cycles.  Consecutive buys or consecutive sells
    (invalid actions that the env penalises) are ignored.
    """
    winning = 0
    total_trades = 0
    hold_times: list[int] = []

    buy_entry: dict | None = None   # step dict at most recent valid buy

    for s in steps:
        if s["action"] == 0 and buy_entry is None:
            # Valid buy (agent wasn't already holding)
            if s["shares"] > 0:          # env accepted the buy
                buy_entry = s
        elif s["action"] == 1 and buy_entry is not None:
            # Valid sell following a buy
            if s["shares"] == 0:         # env accepted the sell
                total_trades += 1
                hold_times.append(s["step"] - buy_entry["step"])
                if s["portfolio_value"] > buy_entry["portfolio_value"]:
                    winning += 1
                buy_entry = None

    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
    avg_hold = (sum(hold_times) / len(hold_times)) if hold_times else 0.0

    return {
        "win_rate": round(win_rate, 1),
        "avg_hold_time": round(avg_hold, 1),
        "total_trades": total_trades,
    }


def _select_action(algo: str, agent, obs: np.ndarray, sim_params: dict) -> int:
    """Apply inference-time simulation parameters to choose an action."""
    from src.agents import device

    algo_up = algo.upper()

    if algo_up == "RANDOM":
        return agent.act(obs)

    if algo_up in ("DQN", "DDQN"):
        epsilon          = float(sim_params.get("epsilon", 0.0))
        action_threshold = float(sim_params.get("action_threshold", 0.0))

        if _random.random() < epsilon:
            return _random.randrange(3)

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(s).squeeze(0)

        best_action = int(q_values.argmax().item())

        # Only trade if the top Q-value beats second by at least action_threshold
        if action_threshold > 0 and best_action != 2:
            sorted_q = q_values.sort(descending=True).values
            if (sorted_q[0] - sorted_q[1]).item() < action_threshold:
                return 2   # hold instead

        return best_action

    if algo_up in ("A2C", "PPO"):
        temperature      = float(sim_params.get("temperature", 1.0))
        min_confidence   = float(sim_params.get("action_threshold", 0.0))

        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h      = agent.model.trunk(s)
            logits = agent.model.actor(h)
            temp   = max(temperature, 1e-3)
            probs  = F.softmax(logits / temp, dim=-1).squeeze(0)

        best_action = int(probs.argmax().item())

        # Only trade if the top probability clears the confidence threshold
        if min_confidence > 0 and best_action != 2:
            if probs[best_action].item() < min_confidence:
                return 2

        return best_action

    # Fallback
    return agent.act_greedy(obs) if hasattr(agent, "act_greedy") else agent.act(obs)


def _run_simulation(algo: str, capital: float, ticker: str = "AAPL",
                    commission: float = 0.0, sim_params: dict | None = None) -> dict:
    from src.env import StockTradingEnv

    sp = sim_params or {}
    position_size = float(sp.get("position_size", 1.0))
    risk_aversion = float(sp.get("risk_aversion", 0.5))

    df = get_df(ticker)
    env = StockTradingEnv(df, train=False, initial_capital=capital,
                          commission_pct=commission,
                          position_size=position_size,
                          risk_aversion=risk_aversion)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = _build_agent(algo, obs_dim, n_actions)

    obs, _ = env.reset()
    done = False
    while not done:
        if sp:
            action = _select_action(algo, agent, obs, sp)
        elif algo.upper() in ("RANDOM", "DQN", "DDQN"):
            action = agent.act(obs)
        else:
            action = agent.act_greedy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    steps = env._step_log
    extra = _compute_extra_metrics(steps, capital)
    return {"steps": steps, "metrics": extra}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    freshness: str | None = None
    available_tickers = []
    for t in TICKERS:
        path = PROC_DIR / f"{t}_features.csv" if t != "AAPL" else DATA_PATH
        if path.exists():
            available_tickers.append(t)

    if DATA_PATH.exists():
        df = get_df("AAPL")
        freshness = str(df["Date"].max().date())

    models_loaded = sum(
        1 for name in ("dqn", "ddqn", "a2c", "ppo")
        if (MODEL_DIR / f"{name}.pth").exists()
    )
    models = {
        name: (MODEL_DIR / f"{name}.pth").exists()
        for name in ("dqn", "ddqn", "a2c", "ppo")
    }
    return {
        "status": "ok",
        "data_loaded": DATA_PATH.exists(),
        "data_freshness": freshness,
        "results_ready": RESULTS_PATH.exists(),
        "models_loaded": models_loaded,
        "models": models,
        "available_tickers": available_tickers,
    }


@app.get("/api/tickers")
def get_tickers():
    available = [
        t for t in TICKERS
        if (PROC_DIR / f"{t}_features.csv").exists()
        or (t == "AAPL" and DATA_PATH.exists())
    ]
    return {"tickers": available, "all": TICKERS}


@app.get("/api/algorithms")
def get_algorithms():
    if not RESULTS_PATH.exists():
        return {"results": [], "buy_and_hold": None}
    data = json.loads(RESULTS_PATH.read_text())
    return data


@app.get("/api/data")
def get_ohlcv(ticker: str = Query(default="AAPL")):
    df = get_df(ticker.upper())
    test = _test_slice(df)
    records = []
    for _, row in test.iterrows():
        records.append({
            "date": str(row["Date"].date()),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return {"ticker": ticker.upper(), "data": records}


@app.get("/api/compare")
def get_compare(ticker: str = Query(default="AAPL")):
    """Run all 5 agents on the specified ticker and return comparison metrics."""
    from src.evaluate import (sharpe_ratio, max_drawdown, total_return_pct,
                               buy_and_hold_return)
    from src.env import StockTradingEnv

    df = get_df(ticker.upper())
    env = StockTradingEnv(df, train=False)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    algos = ["Random", "DQN", "DDQN", "A2C", "PPO"]
    results = []
    for algo in algos:
        try:
            sim = _run_simulation(algo, 100_000.0, ticker.upper())
            steps = sim["steps"]
            hist = [100_000.0] + [s["portfolio_value"] for s in steps]
            results.append({
                "algo": algo,
                "final_portfolio_value": round(hist[-1], 2),
                "total_return_pct": round(total_return_pct(hist), 4),
                "sharpe_ratio": round(sharpe_ratio(hist), 4),
                "max_drawdown": round(max_drawdown(hist), 6),
                "n_steps": len(steps),
                **sim["metrics"],
            })
        except Exception as exc:
            results.append({"algo": algo, "error": str(exc)})

    bah = buy_and_hold_return(df)
    bah_metrics = {
        "algo": "Buy&Hold",
        "final_portfolio_value": round(bah[-1], 2),
        "total_return_pct": round(total_return_pct(bah), 4),
        "sharpe_ratio": round(sharpe_ratio(bah), 4),
        "max_drawdown": round(max_drawdown(bah), 6),
        "n_steps": len(bah) - 1,
    }
    return {"ticker": ticker.upper(), "results": results, "buy_and_hold": bah_metrics}


class SimRequest(BaseModel):
    algo: str
    capital: float = 100_000.0


@app.post("/api/simulate")
def simulate(req: SimRequest):
    result = _run_simulation(req.algo, req.capital)
    return {"algo": req.algo, "capital": req.capital, "steps": result["steps"],
            **result["metrics"]}


class BacktestRequest(BaseModel):
    algo: str
    ticker: str = "AAPL"
    capital: float = 100_000.0
    commission: float = 0.001   # 0.1% per trade side
    # Simulation / inference-time parameters (no retraining needed)
    epsilon: float = 0.0           # DQN/DDQN: exploration rate at inference
    risk_aversion: float = 0.5     # env volatility penalty multiplier
    position_size: float = 1.0     # fraction of capital to deploy per trade
    temperature: float = 1.0       # A2C/PPO: softmax temperature
    action_threshold: float = 0.0  # DQN: min Q-gap to act; A2C/PPO: min prob to trade


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    from src.evaluate import sharpe_ratio, max_drawdown, total_return_pct
    sp = {
        "epsilon":          req.epsilon,
        "risk_aversion":    req.risk_aversion,
        "position_size":    req.position_size,
        "temperature":      req.temperature,
        "action_threshold": req.action_threshold,
    }
    result = _run_simulation(req.algo, req.capital, req.ticker, req.commission, sp)
    steps = result["steps"]
    hist = [req.capital] + [s["portfolio_value"] for s in steps]
    metrics = {
        "final_portfolio_value": round(hist[-1], 2),
        "total_return_pct": round(total_return_pct(hist), 4),
        "sharpe_ratio": round(sharpe_ratio(hist), 4),
        "max_drawdown": round(max_drawdown(hist), 6),
        **result["metrics"],
    }
    return {
        "algo": req.algo,
        "ticker": req.ticker.upper(),
        "capital": req.capital,
        "commission": req.commission,
        "sim_params": sp,
        "steps": steps,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# WebSocket streaming endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        algo       = raw.get("algo", "Random")
        capital    = float(raw.get("capital", 100_000.0))
        speed      = int(raw.get("speed", 1))
        ticker     = raw.get("ticker", "AAPL")
        commission = float(raw.get("commission", 0.0))
        sp = {
            "epsilon":          float(raw.get("epsilon", 0.0)),
            "risk_aversion":    float(raw.get("risk_aversion", 0.5)),
            "position_size":    float(raw.get("position_size", 1.0)),
            "temperature":      float(raw.get("temperature", 1.0)),
            "action_threshold": float(raw.get("action_threshold", 0.0)),
        }

        delay = max(0.005, 0.05 / speed)

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_simulation, algo, capital, ticker, commission, sp
        )
        steps = result["steps"]
        extra = result["metrics"]

        await websocket.send_json({"type": "start", "total_steps": len(steps)})

        for step in steps:
            await websocket.send_json({"type": "step", "data": step})
            await asyncio.sleep(delay)

        await websocket.send_json({"type": "complete", "metrics": extra})
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


@app.websocket("/ws/backtest")
async def ws_backtest(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_json()
        algo = raw.get("algo", "Random")
        capital = float(raw.get("capital", 100_000.0))
        speed = int(raw.get("speed", 1))
        ticker = raw.get("ticker", "AAPL")
        commission = float(raw.get("commission", 0.001))

        delay = max(0.005, 0.05 / speed)

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_simulation, algo, capital, ticker, commission
        )
        steps = result["steps"]
        extra = result["metrics"]

        from src.evaluate import sharpe_ratio, max_drawdown, total_return_pct
        hist = [capital] + [s["portfolio_value"] for s in steps]
        full_metrics = {
            "final_portfolio_value": round(hist[-1], 2),
            "total_return_pct": round(total_return_pct(hist), 4),
            "sharpe_ratio": round(sharpe_ratio(hist), 4),
            "max_drawdown": round(max_drawdown(hist), 6),
            **extra,
        }

        await websocket.send_json({"type": "start", "total_steps": len(steps)})

        for step in steps:
            await websocket.send_json({"type": "step", "data": step})
            await asyncio.sleep(delay)

        await websocket.send_json({"type": "complete", "metrics": full_metrics})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
