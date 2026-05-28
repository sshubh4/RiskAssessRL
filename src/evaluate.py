"""Evaluation metrics and cross-algorithm comparison."""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
DATA_DIR    = pathlib.Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def sharpe_ratio(portfolio_history: list[float] | np.ndarray,
                 risk_free_rate: float = 0.04 / 252) -> float:
    h = np.asarray(portfolio_history, dtype=float)
    returns = np.diff(h) / h[:-1]
    excess = returns - risk_free_rate
    std = excess.std()
    if std < 1e-6:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def max_drawdown(portfolio_history: list[float] | np.ndarray) -> float:
    h = np.asarray(portfolio_history, dtype=float)
    peak = np.maximum.accumulate(h)
    drawdown = (h - peak) / peak
    return float(drawdown.min())


def total_return_pct(portfolio_history: list[float] | np.ndarray) -> float:
    h = np.asarray(portfolio_history, dtype=float)
    return float((h[-1] - h[0]) / h[0] * 100)


def buy_and_hold_return(df: pd.DataFrame, initial_capital: float = 100_000.0) -> list[float]:
    """Kept for backward compatibility — AAPL B&H on the test split."""
    split = int(0.8 * len(df))
    prices = df.iloc[split:]["Close"].values
    shares = initial_capital // prices[0]
    cash = initial_capital - shares * prices[0]
    return [float(cash + shares * p) for p in prices]


# ---------------------------------------------------------------------------
# Benchmark constructors
# ---------------------------------------------------------------------------

def sp500_benchmark(test_prices: np.ndarray, capital: float = 100_000.0) -> dict:
    """
    Primary benchmark: invest same capital in SPY on day 1 of the test period.
    SPY represents the broad market — beating it on a risk-adjusted basis is the
    real goal of a risk-aware RL trading strategy.
    """
    try:
        spy_df = pd.read_csv(DATA_DIR / "SPY_features.csv")
        split  = int(0.8 * len(spy_df))
        spy_prices = spy_df.iloc[split:]["Close"].values
        min_len    = min(len(spy_prices), len(test_prices))
        spy_prices = spy_prices[:min_len]
    except Exception:
        # SPY data unavailable — fall back to the asset itself
        spy_prices = np.asarray(test_prices)

    start  = spy_prices[0]
    shares = int(capital // start)
    cash   = capital - shares * start
    port   = np.array([shares * p + cash for p in spy_prices])

    return {
        "algo":                 "S&P 500",
        "name":                 "S&P 500 (SPY)",
        "final_portfolio_value": round(float(port[-1]), 2),
        "total_return_pct":      round(float((port[-1] - capital) / capital * 100), 4),
        "sharpe_ratio":          round(float(sharpe_ratio(port)), 4),
        "max_drawdown":          round(float(max_drawdown(port)), 6),
        "n_steps":               len(port) - 1,
        "win_rate":              100.0 if port[-1] > capital else 0.0,
        "n_trades":              1,
        "portfolio_history":     port.tolist(),
    }


def aapl_buy_hold_benchmark(test_prices: np.ndarray, capital: float = 100_000.0) -> dict:
    """
    Secondary benchmark: buy the target asset on day 1, hold to end.
    Same start date and capital as all RL agents — a direct apples-to-apples
    comparison of the asset's passive return.
    """
    prices = np.asarray(test_prices)
    start  = prices[0]
    shares = int(capital // start)
    cash   = capital - shares * start
    port   = np.array([shares * p + cash for p in prices])

    return {
        "algo":                 "Buy&Hold",
        "name":                 "Buy & Hold",
        "final_portfolio_value": round(float(port[-1]), 2),
        "total_return_pct":      round(float((port[-1] - capital) / capital * 100), 4),
        "sharpe_ratio":          round(float(sharpe_ratio(port)), 4),
        "max_drawdown":          round(float(max_drawdown(port)), 6),
        "n_steps":               len(port) - 1,
        "win_rate":              100.0 if port[-1] > capital else 0.0,
        "n_trades":              1,
        "portfolio_history":     port.tolist(),
    }


# ---------------------------------------------------------------------------
# Single-agent evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(agent_name: str, agent, env) -> dict:
    obs, _ = env.reset()
    done   = False
    while not done:
        if agent_name in ("DQN", "DDQN", "Random"):
            action = agent.act(obs)
        else:
            action = agent.act_greedy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    hist = env.account_history
    return {
        "algo":                  agent_name,
        "final_portfolio_value": round(hist[-1], 2),
        "total_return_pct":      round(total_return_pct(hist), 4),
        "sharpe_ratio":          round(sharpe_ratio(hist), 4),
        "max_drawdown":          round(max_drawdown(hist), 6),
        "n_steps":               len(hist) - 1,
        "portfolio_history":     [round(v, 2) for v in hist],
    }


# ---------------------------------------------------------------------------
# Full comparison: all agents + both benchmarks → comparison.json
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame, model_dir: pathlib.Path | str = "models",
                   initial_capital: float = 100_000.0) -> dict:
    import torch
    from src.env import StockTradingEnv
    from src.agents import (RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent)

    model_dir  = pathlib.Path(model_dir)
    RESULTS_DIR.mkdir(exist_ok=True)

    def make_env():
        return StockTradingEnv(df, train=False, initial_capital=initial_capital)

    env       = make_env()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    results = []

    # ── RL agents ─────────────────────────────────────────────────────────
    random_agent = RandomAgent(make_env())
    results.append(evaluate_agent("Random", random_agent, make_env()))

    dqn = DQNAgent(obs_dim, n_actions)
    if (model_dir / "dqn.pth").exists():
        dqn.load(str(model_dir / "dqn.pth"))
    results.append(evaluate_agent("DQN", dqn, make_env()))

    ddqn = DoubleDQNAgent(obs_dim, n_actions)
    if (model_dir / "ddqn.pth").exists():
        ddqn.load(str(model_dir / "ddqn.pth"))
    results.append(evaluate_agent("DDQN", ddqn, make_env()))

    a2c = A2CAgent(obs_dim, n_actions)
    if (model_dir / "a2c.pth").exists():
        a2c.load(str(model_dir / "a2c.pth"))
    results.append(evaluate_agent("A2C", a2c, make_env()))

    ppo = PPOAgent(obs_dim, n_actions)
    if (model_dir / "ppo.pth").exists():
        ppo.load(str(model_dir / "ppo.pth"))
    results.append(evaluate_agent("PPO", ppo, make_env()))

    # ── Benchmarks ────────────────────────────────────────────────────────
    split       = int(0.8 * len(df))
    test_prices = df.iloc[split:]["Close"].values

    spy_bench  = sp500_benchmark(test_prices, initial_capital)
    aapl_bench = aapl_buy_hold_benchmark(test_prices, initial_capital)

    # Keep buy_and_hold at top level for backward compat with older frontend code
    out = {
        "results":    results,
        "buy_and_hold": {            # backward compat key
            "algo":                  aapl_bench["algo"],
            "final_portfolio_value": aapl_bench["final_portfolio_value"],
            "total_return_pct":      aapl_bench["total_return_pct"],
            "sharpe_ratio":          aapl_bench["sharpe_ratio"],
            "max_drawdown":          aapl_bench["max_drawdown"],
            "n_steps":               aapl_bench["n_steps"],
        },
        "benchmarks": {
            "sp500":     spy_bench,
            "aapl_hold": aapl_bench,
        },
    }
    out_path = RESULTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[evaluate] saved → {out_path}")
    return out
