"""Training loop for all agents with MLflow experiment tracking."""
from __future__ import annotations
import argparse
import io
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd

from src.env import StockTradingEnv
from src.agents import (RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent)
from src.evaluate import sharpe_ratio, max_drawdown, total_return_pct

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "processed" / "AAPL_features.csv"


def _reward_plot(rewards: list[float], algo: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, linewidth=1.5, color="#00ff87")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{algo} – Reward per Episode"); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = str(MODEL_DIR / f"{algo.lower()}_reward_curve.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _load_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


# ---------------------------------------------------------------------------
# Individual trainers
# ---------------------------------------------------------------------------

def train_dqn(df: pd.DataFrame, episodes: int = 2000, hidden: int = 64,
              gamma: float = 0.99, lr: float = 2.5e-4, algo: str = "DQN") -> DQNAgent:
    AgentCls = DQNAgent if algo == "DQN" else DoubleDQNAgent
    env = StockTradingEnv(df, train=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = AgentCls(obs_dim, n_actions, hidden=hidden, lr=lr, gamma=gamma)

    with mlflow.start_run(run_name=algo):
        mlflow.log_params({
            "algo": algo, "episodes": episodes, "gamma": gamma,
            "lr": lr, "hidden_size": hidden,
        })

        rewards: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset()
            total_r = 0.0
            done = False
            prev_obs = obs
            while not done:
                action = agent.act(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.remember(obs, action, None if done else next_obs, r)
                agent.optimize()
                obs = next_obs
                total_r += r
            agent.episode_end()
            rewards.append(total_r)

            if (ep + 1) % 50 == 0:
                window = rewards[-50:]
                mlflow.log_metrics({
                    "mean_reward": float(np.mean(window)),
                    "portfolio_value": float(env.account_history[-1]),
                }, step=ep + 1)

        # Final test metrics
        test_env = StockTradingEnv(df, train=False)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
        hist = test_env.account_history
        mlflow.log_metrics({
            "final_portfolio_value": hist[-1],
            "total_return_pct": total_return_pct(hist),
            "sharpe_ratio": sharpe_ratio(hist),
            "max_drawdown": max_drawdown(hist),
        })

        MODEL_DIR.mkdir(exist_ok=True)
        save_path = str(MODEL_DIR / f"{algo.lower()}.pth")
        agent.save(save_path)
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(_reward_plot(rewards, algo))

    return agent


def train_a2c(df: pd.DataFrame, episodes: int = 300, hidden: int = 256,
              gamma: float = 0.99, lr: float = 1e-4) -> A2CAgent:
    env = StockTradingEnv(df, train=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = A2CAgent(obs_dim, n_actions, hidden=hidden, lr=lr, gamma=gamma)

    with mlflow.start_run(run_name="A2C"):
        mlflow.log_params({
            "algo": "A2C", "episodes": episodes, "gamma": gamma,
            "lr": lr, "hidden_size": hidden,
        })

        rewards: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            trajectory = []
            total_r = 0.0
            while not done:
                action, value = agent.act(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((obs, action, r, value))
                obs = next_obs
                total_r += r
            agent.update(trajectory)
            rewards.append(total_r)

            if (ep + 1) % 50 == 0:
                window = rewards[-50:]
                mlflow.log_metrics({
                    "mean_reward": float(np.mean(window)),
                    "portfolio_value": float(env.account_history[-1]),
                }, step=ep + 1)

        test_env = StockTradingEnv(df, train=False)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action = agent.act_greedy(obs)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
        hist = test_env.account_history
        mlflow.log_metrics({
            "final_portfolio_value": hist[-1],
            "total_return_pct": total_return_pct(hist),
            "sharpe_ratio": sharpe_ratio(hist),
            "max_drawdown": max_drawdown(hist),
        })

        MODEL_DIR.mkdir(exist_ok=True)
        save_path = str(MODEL_DIR / "a2c.pth")
        agent.save(save_path)
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(_reward_plot(rewards, "A2C"))

    return agent


def train_ppo(df: pd.DataFrame, episodes: int = 300, hidden: int = 256,
              gamma: float = 0.99, lr: float = 1e-4,
              clip: float = 0.2, ppo_epochs: int = 4) -> PPOAgent:
    env = StockTradingEnv(df, train=True)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = PPOAgent(obs_dim, n_actions, hidden=hidden, lr=lr, gamma=gamma,
                     clip=clip, ppo_epochs=ppo_epochs)

    with mlflow.start_run(run_name="PPO"):
        mlflow.log_params({
            "algo": "PPO", "episodes": episodes, "gamma": gamma,
            "lr": lr, "hidden_size": hidden, "clip": clip, "ppo_epochs": ppo_epochs,
        })

        rewards: list[float] = []
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            trajectory = []
            total_r = 0.0
            while not done:
                action, log_prob, value = agent.act(obs)
                next_obs, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((obs, action, r, value, log_prob))
                obs = next_obs
                total_r += r
            agent.update(trajectory)
            rewards.append(total_r)

            if (ep + 1) % 50 == 0:
                window = rewards[-50:]
                mlflow.log_metrics({
                    "mean_reward": float(np.mean(window)),
                    "portfolio_value": float(env.account_history[-1]),
                }, step=ep + 1)

        test_env = StockTradingEnv(df, train=False)
        obs, _ = test_env.reset()
        done = False
        while not done:
            action = agent.act_greedy(obs)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
        hist = test_env.account_history
        mlflow.log_metrics({
            "final_portfolio_value": hist[-1],
            "total_return_pct": total_return_pct(hist),
            "sharpe_ratio": sharpe_ratio(hist),
            "max_drawdown": max_drawdown(hist),
        })

        MODEL_DIR.mkdir(exist_ok=True)
        save_path = str(MODEL_DIR / "ppo.pth")
        agent.save(save_path)
        mlflow.log_artifact(save_path)
        mlflow.log_artifact(_reward_plot(rewards, "PPO"))

    return agent


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL trading agents")
    parser.add_argument("--algo", choices=["DQN", "DDQN", "A2C", "PPO", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--experiment", default="RiskAssessRL")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)
    df = _load_df()

    algos = ["DQN", "DDQN", "A2C", "PPO"] if args.algo == "all" else [args.algo]
    for algo in algos:
        eps = args.episodes
        print(f"\n=== Training {algo} ===")
        if algo == "DQN":
            train_dqn(df, episodes=eps or 2000, algo="DQN")
        elif algo == "DDQN":
            train_dqn(df, episodes=eps or 2000, algo="DDQN")
        elif algo == "A2C":
            train_a2c(df, episodes=eps or 300)
        elif algo == "PPO":
            train_ppo(df, episodes=eps or 300)

    print("\nAll done. Run `mlflow ui` to view results.")


if __name__ == "__main__":
    main()
