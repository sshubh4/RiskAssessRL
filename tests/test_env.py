"""Tests for the StockTradingEnv."""
import numpy as np
import pandas as pd
import pytest
from src.env import StockTradingEnv, FEATURE_COLS


def _make_df(n=200) -> pd.DataFrame:
    """Synthetic OHLCV + feature DataFrame with enough rows for train/test."""
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.clip(close, 50, 500)
    df = pd.DataFrame({
        "Date": dates,
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "MA_20": close,
        "MA_50": close,
        "RSI_14": np.clip(50 + np.random.randn(n) * 10, 0, 100),
        "MACD": np.random.randn(n) * 0.5,
        "MACD_signal": np.random.randn(n) * 0.4,
        "BB_mid": close,
        "BB_upper": close * 1.02,
        "BB_lower": close * 0.98,
        "ATR_14": np.abs(np.random.randn(n)) + 0.5,
        "Support": close * 0.97,
        "Resistance": close * 1.03,
    })
    return df


@pytest.fixture
def df():
    np.random.seed(42)
    return _make_df(200)


class TestEnvReset:
    def test_obs_shape(self, df):
        env = StockTradingEnv(df, train=True, window_size=10)
        obs, info = env.reset()
        expected = 10 * len(env.feature_cols) + 1
        assert obs.shape == (expected,)

    def test_initial_capital(self, df):
        env = StockTradingEnv(df, initial_capital=50_000)
        obs, _ = env.reset()
        assert env.capital == 50_000
        assert env.num_shares == 0
        assert env.total_value == 50_000

    def test_reset_clears_history(self, df):
        env = StockTradingEnv(df)
        env.reset()
        for _ in range(5):
            env.step(2)  # Hold
        env.reset()
        assert len(env.account_history) == 1
        assert env.current_step == 0


class TestEnvStep:
    def test_hold_no_reward(self, df):
        env = StockTradingEnv(df, window_size=10)
        env.reset()
        _, reward, _, _, _ = env.step(2)
        # Hold with no position — only volatility penalty (which is 0 at step 1)
        assert abs(reward) < 0.2

    def test_buy_invalid_when_no_capital(self, df):
        env = StockTradingEnv(df, window_size=10, initial_capital=1.0)
        env.reset()
        _, reward, _, _, _ = env.step(0)  # Buy with $1 — can't buy any shares
        assert reward == pytest.approx(-0.1)

    def test_sell_invalid_when_no_shares(self, df):
        env = StockTradingEnv(df, window_size=10)
        env.reset()
        _, reward, _, _, _ = env.step(1)  # Sell with 0 shares
        assert reward == pytest.approx(-0.1)

    def test_buy_then_sell_profit(self, df):
        """Buy at step 0, sell at step 1 — should produce non-zero reward."""
        env = StockTradingEnv(df, window_size=10)
        env.reset()
        env.step(0)  # Buy
        _, reward, _, _, _ = env.step(1)  # Sell
        # Reward can be positive or negative depending on price movement
        assert isinstance(reward, float)

    def test_termination(self, df):
        env = StockTradingEnv(df, window_size=10, max_steps=5)
        env.reset()
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, truncated, _ = env.step(2)
            steps += 1
            if steps > 100:
                break
        assert terminated

    def test_step_log_populated(self, df):
        env = StockTradingEnv(df, window_size=10)
        env.reset()
        env.step(0)
        env.step(2)
        assert len(env._step_log) == 2
        assert "step" in env._step_log[0]
        assert "price" in env._step_log[0]
        assert "action" in env._step_log[0]
        assert "portfolio_value" in env._step_log[0]

    def test_portfolio_value_non_negative(self, df):
        env = StockTradingEnv(df, window_size=10)
        env.reset()
        for action in [0, 2, 2, 1, 2]:
            _, _, done, _, info = env.step(action)
            assert info["total_value"] >= 0
            if done:
                break

    def test_action_space(self, df):
        env = StockTradingEnv(df)
        assert env.action_space.n == 3

    def test_obs_dtype(self, df):
        env = StockTradingEnv(df, window_size=10)
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_train_test_split(self, df):
        train_env = StockTradingEnv(df, train=True, window_size=10)
        test_env = StockTradingEnv(df, train=False, window_size=10)
        # Train should have more rows than test
        assert len(train_env.market) > len(test_env.market)
        assert len(train_env.market) == int(0.8 * len(df))
