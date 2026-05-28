"""Custom Gymnasium environment for stock trading with risk-adjusted rewards."""
import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_COLS = ["MA_20", "MA_50", "RSI_14", "MACD", "BB_upper", "BB_lower", "ATR_14", "Support", "Resistance"]


class StockTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, train: bool = True, window_size: int = 50,
                 initial_capital: float = 100_000.0, max_steps: int = 150,
                 commission_pct: float = 0.0,
                 position_size: float = 1.0,
                 risk_aversion: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.max_steps = max_steps
        self.commission_pct = commission_pct
        self.position_size = max(0.1, min(1.0, position_size))   # clamp [0.1, 1.0]
        self.risk_aversion = max(0.0, min(1.0, risk_aversion))   # clamp [0.0, 1.0]
        self.feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        split = int(0.8 * len(df))
        self.raw_df = df
        self.market = df.iloc[:split].reset_index(drop=True) if train \
                      else df.iloc[split:].reset_index(drop=True)
        self.train = train

        # Z-score normalise features in-place on the split slice
        self._normed = self.market.copy()
        for col in self.feature_cols:
            mu = self._normed[col].mean()
            sigma = self._normed[col].std()
            if sigma > 0:
                self._normed[col] = (self._normed[col] - mu) / sigma

        self.action_space = spaces.Discrete(3)  # 0=Buy, 1=Sell, 2=Hold
        obs_len = self.window_size * len(self.feature_cols) + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)

        self._step_log: list[dict] = []
        self.reset()

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        window = self._normed.iloc[self.current_step: self.current_step + self.window_size]
        obs = window[self.feature_cols].values.astype(np.float32).flatten()
        holding = np.array([1.0 if self.num_shares > 0 else 0.0], dtype=np.float32)
        return np.concatenate([obs, holding])

    def _price(self) -> float:
        idx = self.current_step + self.window_size
        return float(self.market.iloc[idx]["Close"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.num_shares = 0
        self.book_value = 0.0
        self.total_value = self.initial_capital
        self.account_history: list[float] = [self.initial_capital]
        self.current_step = 0
        self._step_log = []
        return self._obs(), {}

    def step(self, action: int):
        price = self._price()
        reward = 0.0
        invalid = False

        if action == 0:  # Buy — deploy position_size fraction of capital
            deployable = self.capital * self.position_size
            shares = int(deployable // price)
            if shares > 0:
                cost = shares * price
                commission = cost * self.commission_pct
                if cost + commission <= self.capital:
                    self.num_shares += shares
                    self.book_value += cost + commission
                    self.capital -= cost + commission
                else:
                    # Can afford shares but not commission — buy fewer
                    shares = int(self.capital // (price * (1 + self.commission_pct)))
                    if shares > 0:
                        cost = shares * price
                        commission = cost * self.commission_pct
                        self.num_shares += shares
                        self.book_value += cost + commission
                        self.capital -= cost + commission
                    else:
                        invalid = True
            else:
                invalid = True

        elif action == 1:  # Sell — liquidate everything
            if self.num_shares > 0:
                proceeds = self.num_shares * price
                commission = proceeds * self.commission_pct
                net_proceeds = proceeds - commission
                reward += (net_proceeds - self.book_value) / self.initial_capital * 10
                self.capital += net_proceeds
                self.book_value = 0.0
                self.num_shares = 0
            else:
                invalid = True

        if invalid:
            reward -= 0.1

        prev_value = self.total_value
        self.total_value = self.capital + self.num_shares * price
        self.account_history.append(self.total_value)

        # Holding reward: small positive signal for growing portfolio
        if self.total_value > prev_value:
            reward += 0.01

        # Sharpe-inspired volatility penalty — scaled by risk_aversion
        # risk_aversion=0.0 → no penalty; 0.5 → baseline 0.1; 1.0 → 0.2
        if len(self.account_history) > 10:
            hist = np.array(self.account_history[-11:])
            rets = np.diff(hist) / hist[:-1]
            reward -= np.std(rets) * 0.1 * (self.risk_aversion * 2.0)

        self._step_log.append({
            "step": self.current_step,
            "price": round(price, 4),
            "action": int(action),
            "portfolio_value": round(self.total_value, 2),
            "capital": round(self.capital, 2),
            "shares": int(self.num_shares),
        })

        self.current_step += 1
        terminated = (self.current_step + self.window_size) >= len(self.market) - 1 \
                     or self.current_step >= self.max_steps
        return self._obs(), reward, terminated, False, {
            "total_value": self.total_value,
            "capital": self.capital,
            "shares": self.num_shares,
            "price": price,
        }

    def render(self):
        plt.figure(figsize=(14, 5))
        plt.plot(self.account_history, color="#00ff87", linewidth=2, label="Portfolio")
        plt.axhline(self.initial_capital, color="red", linestyle="--", label="Initial capital")
        plt.xlabel("Step"); plt.ylabel("Value ($)")
        plt.title("Portfolio Value"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()
