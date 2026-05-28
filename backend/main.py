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


def _run_simulation(
    algo: str,
    capital: float,
    ticker: str = "AAPL",
    commission: float = 0.0,
    sim_params: dict | None = None,
    stop_loss: float | None = None,          # fraction, e.g. 0.10 = stop at -10%
    take_profit: float | None = None,        # fraction, e.g. 0.20 = exit at +20%
    max_drawdown_kill: float | None = None,  # fraction, e.g. 0.20 = halt at -20% from peak
    test_start_idx: int | None = None,       # bar index in full df where test begins
    test_end_idx: int | None = None,         # bar index in full df where test ends (exclusive)
) -> dict:
    from src.env import StockTradingEnv

    sp = sim_params or {}
    position_size = float(sp.get("position_size", 1.0))
    risk_aversion = float(sp.get("risk_aversion", 0.5))

    df            = get_df(ticker)
    default_split = int(0.8 * len(df))
    custom_split  = test_start_idx if test_start_idx is not None else default_split
    custom_end    = test_end_idx   if test_end_idx   is not None else len(df)
    # Clamp to valid bounds
    custom_split  = max(0, min(custom_split, len(df) - 21))
    custom_end    = max(custom_split + 20, min(custom_end, len(df)))

    test_df  = df.iloc[custom_split:custom_end].reset_index(drop=True)  # for price lookups

    env = StockTradingEnv(df, train=False, initial_capital=capital,
                          commission_pct=commission,
                          position_size=position_size,
                          risk_aversion=risk_aversion,
                          custom_split=custom_split,
                          custom_end=custom_end)
    obs_dim  = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent    = _build_agent(algo, obs_dim, n_actions)

    obs, _ = env.reset()
    done   = False

    # ── Risk-management tracking ─────────────────────────────────────────────
    in_position  = False
    entry_price: float | None = None
    peak_value   = capital
    forced_stop  = False
    stop_reason: str | None = None

    while not done:
        # Current price (before this step)
        cur_step = getattr(env, "current_step", 0)
        current_price: float | None = (
            float(test_df.iloc[cur_step]["Close"])
            if cur_step < len(test_df) else None
        )

        # Choose action
        if sp:
            action = _select_action(algo, agent, obs, sp)
        elif algo.upper() in ("RANDOM", "DQN", "DDQN"):
            action = agent.act(obs)
        else:
            action = agent.act_greedy(obs)

        # ── Override action if stop-loss / take-profit triggered ─────────────
        if in_position and entry_price is not None and current_price is not None:
            price_ret = (current_price - entry_price) / entry_price
            if stop_loss and price_ret <= -abs(stop_loss):
                action = 1   # forced sell — stop loss
            elif take_profit and price_ret >= abs(take_profit):
                action = 1   # forced sell — take profit

        obs, _, terminated, truncated, _ = env.step(action)

        # ── Update position tracking ─────────────────────────────────────────
        if env._step_log:
            last = env._step_log[-1]
            if last["action"] == 0 and last.get("shares", 0) > 0 and not in_position:
                in_position = True
                entry_price = current_price
            elif last["action"] == 1 and last.get("shares", 0) == 0 and in_position:
                in_position = False
                entry_price = None

            # ── Max-drawdown kill switch ─────────────────────────────────────
            if max_drawdown_kill:
                pv = last["portfolio_value"]
                peak_value = max(peak_value, pv)
                if (pv - peak_value) / peak_value <= -abs(max_drawdown_kill):
                    forced_stop = True
                    stop_reason = "max_drawdown_kill"
                    break

        done = terminated or truncated

    steps = env._step_log
    extra = _compute_extra_metrics(steps, capital)
    if forced_stop:
        extra["forced_stop"] = True
        extra["stop_reason"] = stop_reason
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


def _compute_regime() -> dict:
    """Detect market regime from SPY test-period return."""
    try:
        spy_path = PROC_DIR / "SPY_features.csv"
        spy_df   = pd.read_csv(spy_path, parse_dates=["Date"])
        split    = int(0.8 * len(spy_df))
        spy_test = spy_df.iloc[split:].reset_index(drop=True)
        spy_ret  = (spy_test["Close"].iloc[-1] - spy_test["Close"].iloc[0]) / spy_test["Close"].iloc[0] * 100
        spy_ret  = round(float(spy_ret), 1)
        if spy_ret > 15:
            return {"label": "Bull Market",     "color": "#26a69a", "spy_return": spy_ret}
        if spy_ret < -10:
            return {"label": "Bear Market",     "color": "#ef5350", "spy_return": spy_ret}
        return     {"label": "Sideways Market", "color": "#f5c842", "spy_return": spy_ret}
    except Exception:
        return {"label": "Unknown", "color": "#787b86", "spy_return": 0}


@app.get("/api/algorithms")
def get_algorithms():
    if not RESULTS_PATH.exists():
        return {"results": [], "buy_and_hold": None,
                "benchmarks": {"sp500": {}, "aapl_hold": {}}}
    data = json.loads(RESULTS_PATH.read_text())
    # Ensure benchmarks key exists (old comparison.json may not have it)
    if "benchmarks" not in data:
        data["benchmarks"] = {"sp500": {}, "aapl_hold": {}}
    return data


@app.get("/api/data")
def get_ohlcv(ticker: str = Query(default="AAPL")):
    df = get_df(ticker.upper())
    test_start_idx = int(0.8 * len(df))   # index in full array where test period begins
    records = []
    for _, row in df.iterrows():           # return ALL bars (train + test)
        records.append({
            "date":   str(row["Date"].date()),
            "open":   round(float(row["Open"]),   4),
            "high":   round(float(row["High"]),   4),
            "low":    round(float(row["Low"]),    4),
            "close":  round(float(row["Close"]),  4),
            "volume": int(row["Volume"]),
        })
    regime = _compute_regime()
    return {
        "ticker":         ticker.upper(),
        "data":           records,
        "test_start_idx": test_start_idx,
        "regime":         regime,
    }


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

    from src.evaluate import sp500_benchmark, aapl_buy_hold_benchmark
    split       = int(0.8 * len(df))
    test_prices = df.iloc[split:]["Close"].values
    spy_bench   = sp500_benchmark(test_prices)
    aapl_bench  = aapl_buy_hold_benchmark(test_prices)
    return {
        "ticker":      ticker.upper(),
        "results":     results,
        "buy_and_hold": aapl_bench,         # backward compat
        "benchmarks":  {"sp500": spy_bench, "aapl_hold": aapl_bench},
    }


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


class RunAllRequest(BaseModel):
    ticker:             str   = "AAPL"
    capital:            float = 100_000.0
    commission:         float = 0.001
    position_size:      float = 1.0
    stop_loss:          float | None = None   # fraction e.g. 0.10
    take_profit:        float | None = None   # fraction e.g. 0.20
    max_drawdown_kill:  float | None = None   # fraction e.g. 0.20
    test_start_idx:     int   | None = None   # bar index in full df
    test_end_idx:       int   | None = None   # bar index in full df (exclusive)


@app.post("/api/run_all_backtest")
def run_all_backtest(req: RunAllRequest):
    """Run all 5 algorithms with the same backtest parameters and return comparison."""
    from src.evaluate import sharpe_ratio, max_drawdown, total_return_pct
    from src.evaluate import sp500_benchmark, aapl_buy_hold_benchmark

    sp = {"position_size": req.position_size, "risk_aversion": 0.5,
          "epsilon": 0.0, "temperature": 1.0, "action_threshold": 0.0}

    results = []
    for algo in ["DQN", "DDQN", "A2C", "PPO", "Random"]:
        try:
            sim  = _run_simulation(
                algo, req.capital, req.ticker, req.commission, sp,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
                max_drawdown_kill=req.max_drawdown_kill,
                test_start_idx=req.test_start_idx,
                test_end_idx=req.test_end_idx,
            )
            hist = [req.capital] + [s["portfolio_value"] for s in sim["steps"]]
            results.append({
                "algo":                  algo,
                "name":                  algo,
                "final_portfolio_value": round(hist[-1], 2),
                "total_return_pct":      round(total_return_pct(hist), 4),
                "sharpe_ratio":          round(sharpe_ratio(hist), 4),
                "max_drawdown":          round(max_drawdown(hist), 6),
                "n_trades":              sim["metrics"]["total_trades"],
                "win_rate":              sim["metrics"]["win_rate"],
                "portfolio_history":     hist,
                **sim["metrics"],
            })
        except Exception as exc:
            results.append({"algo": algo, "error": str(exc)})

    df           = get_df(req.ticker)
    default_split = int(0.8 * len(df))
    t_start      = req.test_start_idx if req.test_start_idx is not None else default_split
    t_end        = req.test_end_idx   if req.test_end_idx   is not None else len(df)
    test_prices  = df.iloc[t_start:t_end]["Close"].values
    spy_bench   = sp500_benchmark(test_prices,          capital=req.capital)
    aapl_bench  = aapl_buy_hold_benchmark(test_prices,  capital=req.capital)

    return {
        "results":     results,
        "buy_and_hold": aapl_bench,
        "benchmarks":  {"sp500": spy_bench, "aapl_hold": aapl_bench},
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
        # Risk-management overrides (backtest mode)
        raw_sl  = raw.get("stop_loss",         None)
        raw_tp  = raw.get("take_profit",        None)
        raw_mdd = raw.get("max_drawdown_kill",  None)
        stop_loss         = float(raw_sl)  if raw_sl  is not None else None
        take_profit       = float(raw_tp)  if raw_tp  is not None else None
        max_drawdown_kill = float(raw_mdd) if raw_mdd is not None else None
        # Custom test date range
        raw_tsi = raw.get("test_start_idx", None)
        raw_tei = raw.get("test_end_idx",   None)
        test_start_idx = int(raw_tsi) if raw_tsi is not None else None
        test_end_idx   = int(raw_tei) if raw_tei is not None else None

        delay = max(0.005, 0.05 / speed)

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_simulation, algo, capital, ticker, commission, sp,
            stop_loss, take_profit, max_drawdown_kill,
            test_start_idx, test_end_idx,
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
