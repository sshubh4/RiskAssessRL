"""Evaluation metrics and cross-algorithm comparison."""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def sharpe_ratio(portfolio_history: list[float] | np.ndarray,
                 risk_free_rate: float = 0.04 / 252) -> float:
    h = np.asarray(portfolio_history, dtype=float)
    returns = np.diff(h) / h[:-1]
    excess = returns - risk_free_rate
    std = excess.std()
    # Degenerate: agent never traded — portfolio is a flat line, std ≈ 0
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
    """Compute buy-and-hold portfolio value for the test split."""
    split = int(0.8 * len(df))
    prices = df.iloc[split:]["Close"].values
    shares = initial_capital // prices[0]
    cash = initial_capital - shares * prices[0]
    return [float(cash + shares * p) for p in prices]


# ---------------------------------------------------------------------------
# Run one agent on the test env and collect metrics
# ---------------------------------------------------------------------------

def evaluate_agent(agent_name: str, agent, env) -> dict:
    obs, _ = env.reset()
    done = False
    while not done:
        if agent_name == "Random":
            action = agent.act(obs)
        elif agent_name in ("DQN", "DDQN"):
            action = agent.act(obs)
        else:
            action = agent.act_greedy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    hist = env.account_history
    return {
        "algo": agent_name,
        "final_portfolio_value": round(hist[-1], 2),
        "total_return_pct": round(total_return_pct(hist), 4),
        "sharpe_ratio": round(sharpe_ratio(hist), 4),
        "max_drawdown": round(max_drawdown(hist), 6),
        "n_steps": len(hist) - 1,
    }


# ---------------------------------------------------------------------------
# Run all agents and save comparison.json
# ---------------------------------------------------------------------------

def run_comparison(df: pd.DataFrame, model_dir: pathlib.Path | str = "models",
                   initial_capital: float = 100_000.0) -> dict:
    import torch
    from src.env import StockTradingEnv
    from src.agents import (RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent)

    model_dir = pathlib.Path(model_dir)
    RESULTS_DIR.mkdir(exist_ok=True)

    def make_env():
        return StockTradingEnv(df, train=False, initial_capital=initial_capital)

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    results = []

    # Random
    random_agent = RandomAgent(make_env())
    results.append(evaluate_agent("Random", random_agent, make_env()))

    # DQN
    dqn = DQNAgent(obs_dim, n_actions)
    if (model_dir / "dqn.pth").exists():
        dqn.load(str(model_dir / "dqn.pth"))
    results.append(evaluate_agent("DQN", dqn, make_env()))

    # DDQN
    ddqn = DoubleDQNAgent(obs_dim, n_actions)
    if (model_dir / "ddqn.pth").exists():
        ddqn.load(str(model_dir / "ddqn.pth"))
    results.append(evaluate_agent("DDQN", ddqn, make_env()))

    # A2C
    a2c = A2CAgent(obs_dim, n_actions)
    if (model_dir / "a2c.pth").exists():
        a2c.load(str(model_dir / "a2c.pth"))
    results.append(evaluate_agent("A2C", a2c, make_env()))

    # PPO
    ppo = PPOAgent(obs_dim, n_actions)
    if (model_dir / "ppo.pth").exists():
        ppo.load(str(model_dir / "ppo.pth"))
    results.append(evaluate_agent("PPO", ppo, make_env()))

    # Buy-and-hold benchmark
    bah = buy_and_hold_return(df, initial_capital)
    bah_metrics = {
        "algo": "Buy&Hold",
        "final_portfolio_value": round(bah[-1], 2),
        "total_return_pct": round(total_return_pct(bah), 4),
        "sharpe_ratio": round(sharpe_ratio(bah), 4),
        "max_drawdown": round(max_drawdown(bah), 6),
        "n_steps": len(bah) - 1,
    }
    results.append(bah_metrics)

    out = {"results": results, "buy_and_hold": bah_metrics}
    out_path = RESULTS_DIR / "comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[evaluate] saved → {out_path}")
    return out
