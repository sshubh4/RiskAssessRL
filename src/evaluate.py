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
    # Policy-gradient agents were trained with action masking, so they must be
    # evaluated with it too (otherwise the argmax can pick an action the policy
    # never learned to take when invalid). DQN/DDQN were trained unmasked.
    use_mask = agent_name in ("A2C", "PPO")
    while not done:
        # Random is the only agent that should act stochastically at eval time.
        # Every trained agent acts greedily — for DQN/DDQN this means act_greedy,
        # NOT act() (which is epsilon-greedy and ~random on a freshly loaded net).
        if agent_name == "Random":
            action = agent.act(obs)
        else:
            mask = env.valid_action_mask() if use_mask else None
            action = agent.act_greedy(obs, mask)
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


# ---------------------------------------------------------------------------
# Lightweight, MLflow-free trainers (used by walk-forward analysis)
# ---------------------------------------------------------------------------

def _train_value_agent(df, algo, custom_split, episodes):
    """Train a fresh DQN/DDQN on df[:custom_split]. No MLflow, no disk I/O."""
    from src.env import StockTradingEnv
    from src.agents import DQNAgent, DoubleDQNAgent
    env = StockTradingEnv(df, train=True, custom_split=custom_split)
    Cls = DQNAgent if algo == "DQN" else DoubleDQNAgent
    agent = Cls(env.observation_space.shape[0], env.action_space.n)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = agent.act(obs)
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(obs, a, None if done else nobs, r)
            agent.optimize()
            obs = nobs
        agent.episode_end()
    return agent


def _train_a2c_agent(df, custom_split, episodes):
    from src.env import StockTradingEnv
    from src.agents import A2CAgent
    env = StockTradingEnv(df, train=True, custom_split=custom_split)
    agent = A2CAgent(env.observation_space.shape[0], env.action_space.n)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        traj = []
        while not done:
            m = env.valid_action_mask()
            a, v = agent.act(obs, m)
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            traj.append((obs, a, r, v, m))
            obs = nobs
        agent.update(traj, entropy_coef=0.05 - 0.045 * ep / max(1, episodes - 1))
    return agent


def _train_ppo_agent(df, custom_split, episodes):
    from src.env import StockTradingEnv
    from src.agents import PPOAgent
    env = StockTradingEnv(df, train=True, custom_split=custom_split)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        traj = []
        while not done:
            m = env.valid_action_mask()
            a, lp, v = agent.act(obs, m)
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            traj.append((obs, a, r, v, lp, m))
            obs = nobs
        agent.update(traj, entropy_coef=0.05 - 0.045 * ep / max(1, episodes - 1))
    return agent


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward(df: pd.DataFrame, n_folds: int = 5,
                 value_episodes: int = 600, policy_episodes: int = 600,
                 initial_capital: float = 100_000.0) -> dict:
    """
    Expanding-window walk-forward backtest.

    The data is cut into ``n_folds + 1`` equal blocks. Fold k trains on every
    block up to and including block k, then tests on block k+1:

        fold 0:  train [block 0]            test [block 1]
        fold 1:  train [block 0..1]         test [block 2]
        ...
        fold 4:  train [block 0..4]         test [block 5]

    Each fold trains FRESH agents (no leakage from later data into earlier
    folds) and evaluates them greedily on the held-out next block. We then
    report mean ± std of return and Sharpe across folds — a far more honest
    measure of regime generalisation than a single fixed 80/20 split.
    """
    from src.env import StockTradingEnv
    from src.agents import RandomAgent

    N = len(df)
    block = N // (n_folds + 1)
    algos = ["Random", "DQN", "DDQN", "A2C", "PPO"]
    per_fold: dict[str, list[dict]] = {a: [] for a in algos}
    folds_meta = []

    has_date = "Date" in df.columns

    for k in range(n_folds):
        train_end  = (k + 1) * block
        test_start = train_end
        test_end   = (k + 2) * block if k < n_folds - 1 else N

        def make_test_env():
            return StockTradingEnv(df, train=False, initial_capital=initial_capital,
                                   custom_split=test_start, custom_end=test_end)

        meta = {
            "fold":        k + 1,
            "train_rows":  train_end,
            "test_rows":   test_end - test_start,
        }
        if has_date:
            meta["train_dates"] = [str(df.iloc[0]["Date"].date()),
                                   str(df.iloc[train_end - 1]["Date"].date())]
            meta["test_dates"]  = [str(df.iloc[test_start]["Date"].date()),
                                   str(df.iloc[test_end - 1]["Date"].date())]
        folds_meta.append(meta)
        print(f"\n[walk-forward] fold {k+1}/{n_folds}  "
              f"train rows 0..{train_end}  test rows {test_start}..{test_end}")

        # ── Random baseline (no training) ──
        per_fold["Random"].append(
            evaluate_agent("Random", RandomAgent(make_test_env()), make_test_env()))

        # ── Trained agents ──
        print("  training DQN…")
        dqn = _train_value_agent(df, "DQN", train_end, value_episodes)
        per_fold["DQN"].append(evaluate_agent("DQN", dqn, make_test_env()))

        print("  training DDQN…")
        ddqn = _train_value_agent(df, "DDQN", train_end, value_episodes)
        per_fold["DDQN"].append(evaluate_agent("DDQN", ddqn, make_test_env()))

        print("  training A2C…")
        a2c = _train_a2c_agent(df, train_end, policy_episodes)
        per_fold["A2C"].append(evaluate_agent("A2C", a2c, make_test_env()))

        print("  training PPO…")
        ppo = _train_ppo_agent(df, train_end, policy_episodes)
        per_fold["PPO"].append(evaluate_agent("PPO", ppo, make_test_env()))

    # ── Aggregate mean ± std per algorithm ──
    summary = []
    for algo in algos:
        rets    = np.array([f["total_return_pct"] for f in per_fold[algo]], dtype=float)
        sharpes = np.array([f["sharpe_ratio"]     for f in per_fold[algo]], dtype=float)
        dds     = np.array([f["max_drawdown"]     for f in per_fold[algo]], dtype=float)
        summary.append({
            "algo":           algo,
            "return_mean":    round(float(rets.mean()), 4),
            "return_std":     round(float(rets.std()), 4),
            "sharpe_mean":    round(float(sharpes.mean()), 4),
            "sharpe_std":     round(float(sharpes.std()), 4),
            "max_dd_mean":    round(float(dds.mean()), 6),
            "fold_returns":   [round(float(x), 4) for x in rets],
            "fold_sharpes":   [round(float(x), 4) for x in sharpes],
        })

    out = {
        "method":      f"expanding-window walk-forward, {n_folds} folds",
        "value_episodes_per_fold":  value_episodes,
        "policy_episodes_per_fold": policy_episodes,
        "folds":       folds_meta,
        "summary":     summary,
        "per_fold":    {a: [{k: v for k, v in f.items() if k != "portfolio_history"}
                            for f in per_fold[a]] for a in algos},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "walkforward.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[walk-forward] saved → {out_path}")
    return out


# ---------------------------------------------------------------------------
# Zero-shot cross-asset generalisation
# ---------------------------------------------------------------------------

def generalization(model_dir: pathlib.Path | str = "models",
                   train_ticker: str = "AAPL",
                   tickers: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA", "SPY"),
                   initial_capital: float = 100_000.0) -> dict:
    """
    Take the DDQN agent trained ONLY on ``train_ticker`` and run it zero-shot
    on the held-out test slice of every ticker. No retraining, no fine-tuning.
    A small degradation on unseen tickers means the agent learned a transferable
    pattern; a collapse means it overfit to one asset's idiosyncrasies.
    """
    from src.env import StockTradingEnv
    from src.agents import DoubleDQNAgent

    model_dir = pathlib.Path(model_dir)
    weights = model_dir / "ddqn.pth"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found — train DDQN first.")

    # Build agent with the obs dim of the training ticker, load once, reuse.
    train_df = pd.read_csv(DATA_DIR / f"{train_ticker}_features.csv", parse_dates=["Date"])
    ref_env  = StockTradingEnv(train_df, train=False, initial_capital=initial_capital)
    obs_dim  = ref_env.observation_space.shape[0]
    ddqn = DoubleDQNAgent(obs_dim, ref_env.action_space.n)
    ddqn.load(str(weights))

    rows = []
    for tk in tickers:
        path = DATA_DIR / f"{tk}_features.csv"
        if not path.exists():
            print(f"[generalization] skip {tk} (no data)")
            continue
        df_t = pd.read_csv(path, parse_dates=["Date"])
        env  = StockTradingEnv(df_t, train=False, initial_capital=initial_capital)
        if env.observation_space.shape[0] != obs_dim:
            print(f"[generalization] skip {tk} (obs dim mismatch)")
            continue

        res = evaluate_agent("DDQN", ddqn, env)

        # Buy & hold on the same test slice, for context
        split = int(0.8 * len(df_t))
        bh = aapl_buy_hold_benchmark(df_t.iloc[split:]["Close"].values, initial_capital)

        rows.append({
            "ticker":            tk,
            "is_train_ticker":   tk == train_ticker,
            "ddqn_return_pct":   res["total_return_pct"],
            "ddqn_sharpe":       res["sharpe_ratio"],
            "ddqn_max_drawdown": res["max_drawdown"],
            "buy_hold_return_pct": bh["total_return_pct"],
            "alpha_vs_buy_hold":   round(res["total_return_pct"] - bh["total_return_pct"], 4),
            "n_steps":           res["n_steps"],
        })
        print(f"[generalization] {tk:5s}  return {res['total_return_pct']:+7.2f}%  "
              f"sharpe {res['sharpe_ratio']:+.3f}  (B&H {bh['total_return_pct']:+.2f}%)")

    out = {
        "train_ticker": train_ticker,
        "description":  f"DDQN trained only on {train_ticker}, evaluated zero-shot "
                        f"on the test slice of each ticker (no retraining).",
        "results":      rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "generalization.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[generalization] saved → {out_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL trading agents")
    parser.add_argument("--mode", choices=["comparison", "walkforward", "generalization"],
                        default="comparison")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--value-episodes",  type=int, default=600)
    parser.add_argument("--policy-episodes", type=int, default=600)
    args = parser.parse_args()

    if args.mode == "generalization":
        generalization(train_ticker=args.ticker)
        return

    df = pd.read_csv(DATA_DIR / f"{args.ticker}_features.csv", parse_dates=["Date"])
    if args.mode == "walkforward":
        walk_forward(df, n_folds=args.folds,
                     value_episodes=args.value_episodes,
                     policy_episodes=args.policy_episodes)
    else:
        run_comparison(df, model_dir="models")


if __name__ == "__main__":
    main()
