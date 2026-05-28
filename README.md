# RiskAssessRL

> Production-grade reinforcement learning platform for multi-asset, risk-adjusted trading strategy simulation — with a TradingView-style dashboard.

[![Dashboard](https://img.shields.io/badge/Dashboard-localhost%3A3000-2962ff?style=flat-square)](http://localhost:3000)
[![MLflow](https://img.shields.io/badge/MLflow_UI-localhost%3A5000-0194e2?style=flat-square)](http://localhost:5000)
[![API Docs](https://img.shields.io/badge/API_Docs-localhost%3A8000%2Fdocs-009688?style=flat-square)](http://localhost:8000/docs)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-44%20passed-26a69a?style=flat-square)](tests/)

---

## What It Does

Five reinforcement learning agents (Random, DQN, DDQN, A2C, PPO) trade a configurable portfolio against historical stock data. A live dashboard streams trades step-by-step on an interactive TradingView-style candlestick chart, plots portfolio vs buy-and-hold, and displays Sharpe ratio, max drawdown, win rate, and alpha — in real time.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        docker-compose                          │
│                                                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │    frontend      │  │     backend      │  │   mlflow    │  │
│  │  React + Vite    │  │    FastAPI        │  │  tracking   │  │
│  │  :3000           │  │    :8000         │  │  :5000      │  │
│  │                  │  │                  │  │             │  │
│  │  lightweight-    │  │  REST endpoints  │  │  mlruns/    │  │
│  │  charts (TV)     │  │  WebSocket sim   │  │             │  │
│  │  Tailwind CSS    │  │  RL agents       │  │             │  │
│  └────────┬─────────┘  └────────┬─────────┘  └─────────────┘  │
│           │   WebSocket /ws/sim  │                              │
│           └─────────────────────┘                              │
└────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴──────────────────┐
          │             pipeline/                 │
          │  ingest → features → validate         │
          │  • 5 tickers: AAPL MSFT GOOGL NVDA SPY│
          │  • Parallel ThreadPoolExecutor        │
          │  • APScheduler: daily 18:00 ET        │
          └──────────────────────────────────────┘
                              │
          ┌───────────────────┴──────────────────┐
          │  data/                                │
          │  raw/{TICKER}.csv                     │
          │  processed/{TICKER}_features.csv      │
          └──────────────────────────────────────┘
```

---

## Quickstart

### Local (dev mode — fastest)

```bash
# 1. Create and activate virtual environment
python3 -m venv ~/myenv && source ~/myenv/bin/activate

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run the multi-asset data pipeline (downloads AAPL MSFT GOOGL NVDA SPY in parallel)
PYTHONPATH=. python -m pipeline.ingest

# 4. Train all agents in parallel (10K eps for DQN/DDQN, 3K for A2C/PPO)
PYTHONPATH=. python -m src.train --algo DQN  --episodes 10000 &
PYTHONPATH=. python -m src.train --algo DDQN --episodes 10000 &
PYTHONPATH=. python -m src.train --algo A2C  --episodes 3000  &
PYTHONPATH=. python -m src.train --algo PPO  --episodes 3000  &
wait

# 5. Evaluate and generate comparison.json
PYTHONPATH=. python -c "
import pandas as pd; from src.evaluate import run_comparison
df = pd.read_csv('data/processed/AAPL_features.csv', parse_dates=['Date'])
run_comparison(df, model_dir='models')
"

# 6. Start backend
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000** to see the dashboard.

### Docker

```bash
# After running steps 1-5 above to populate data/ and models/:
docker-compose up --build
```

Services:
- Dashboard: http://localhost:3000
- API + Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## Project Structure

```
RiskAssessRL/
├── pipeline/
│   ├── ingest.py        # yfinance download → data/raw/; run_all() for 5 tickers in parallel
│   ├── features.py      # MA20/50, RSI-14, MACD, Bollinger Bands, ATR-14, Support/Resistance
│   ├── validate.py      # Pandera schema (nulls, price > 0, RSI 0-100, monotonic dates)
│   └── scheduler.py     # APScheduler BlockingScheduler — runs daily at 18:00 ET
├── src/
│   ├── env.py           # StockTradingEnv: 50-day window, 9 features, Discrete(3) + commission
│   ├── agents.py        # RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent
│   ├── train.py         # MLflow-wrapped training; CLI: python -m src.train --algo X --episodes N
│   └── evaluate.py      # sharpe_ratio, max_drawdown, total_return_pct, run_comparison()
├── backend/
│   └── main.py          # FastAPI: REST + WebSocket; multi-ticker; commission; win rate
├── frontend/
│   └── src/
│       ├── App.jsx                   # Root: WebSocket state, BAH computation, layout
│       └── components/
│           ├── Topbar.jsx            # 48px topbar: ticker pills, price, clock, status
│           ├── LeftPanel.jsx         # Strategy selector, backtest params, live stats
│           ├── TradingChart.jsx      # lightweight-charts CandlestickSeries + vol + markers
│           ├── PortfolioChart.jsx    # lightweight-charts LineSeries — portfolio vs B&H
│           ├── RightPanel.jsx        # Price card, portfolio card, Sharpe/DD/WR, vs B&H bars
│           └── ComparisonTable.jsx   # All-algo table with best-value highlighting
├── tests/
│   ├── test_env.py        # 10 tests: obs shape, capital, actions, step log, split
│   ├── test_pipeline.py   # 17 tests: indicators, schema validation, edge cases
│   └── test_api.py        # 17 tests: health, algorithms, data, simulate endpoints
├── data/
│   ├── raw/               # Raw OHLCV CSVs per ticker
│   └── processed/         # Feature-engineered CSVs per ticker
├── models/                # Trained .pth weight files
├── results/               # comparison.json
├── mlruns/                # MLflow experiment artifacts (excluded from git)
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

---

## Algorithms

| Algorithm | Type | Replay | Policy | Network |
|---|---|---|---|---|
| **Random** | Baseline | — | Uniform random | — |
| **DQN** | Value-based | ✓ (10K) | ε-greedy → greedy | 2-layer MLP (64) |
| **DDQN** | Value-based | ✓ (10K) | ε-greedy → greedy | 2-layer MLP (64), decoupled target |
| **A2C** | Policy gradient | — | Stochastic (train) / greedy (eval) | Actor-Critic (256) |
| **PPO** | Policy gradient | — | Stochastic (train) / greedy (eval) | Actor-Critic (256), clipped ratio |

**Observation**: 50-day window × 9 Z-scored features + 1 holding flag = **451-dim** vector
**Actions**: `0` = Buy all-in, `1` = Sell all, `2` = Hold
**Reward**: `profit / initial_capital × 10` − `volatility × 0.1` + `0.01` if growing − `0.1` for invalid

---

## Algorithm Comparison (AAPL, 20% test split)

| Algorithm | Final Value | Return | Sharpe | Max DD | Trades | Win Rate |
|---|---|---|---|---|---|---|
| Random | $104,700 | +4.70% | 0.330 | −4.2% | ~22 | ~45% |
| DQN | $101,440 | +1.44% | −0.012 | −6.1% | — | — |
| DDQN | $93,860 | −6.14% | −0.930 | −8.5% | — | — |
| A2C | $100,000 | 0.00% | 0.000 | 0.0% | 0 | — |
| PPO | $100,000 | 0.00% | 0.000 | 0.0% | 0 | — |
| **Buy & Hold** | **$152,000** | **+52.0%** | **1.896** | **−9.1%** | — | — |

*Results from 300-episode pre-training (before the production 10K-episode runs complete).*

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status, model count, data freshness, available tickers |
| `GET` | `/api/tickers` | List of available tickers |
| `GET` | `/api/algorithms` | All-algo metrics from `results/comparison.json` |
| `GET` | `/api/data?ticker=AAPL` | OHLCV for the test period of any ticker |
| `POST` | `/api/simulate` | `{algo, capital}` → full step log + win rate + avg hold time |
| `POST` | `/api/backtest` | `{algo, ticker, capital, commission}` → steps + metrics |
| `GET` | `/api/compare?ticker=AAPL` | Live comparison of all 5 agents on a ticker |
| `WS` | `/ws/simulate` | Real-time step streaming with configurable speed + complete event |
| `WS` | `/ws/backtest` | Same as above, with commission and full metrics on completion |

Interactive docs: **http://localhost:8000/docs**

---

## Pipeline

```bash
# Single ticker (AAPL)
PYTHONPATH=. python -m pipeline.ingest --ticker AAPL

# All 5 tickers in parallel (default)
PYTHONPATH=. python -m pipeline.ingest

# Start the scheduler (runs daily at 18:00 ET automatically)
PYTHONPATH=. python -m pipeline.scheduler
```

---

## MLflow Experiment Tracking

Each `src.train` run logs to MLflow automatically:

| Category | Items |
|---|---|
| **Parameters** | `algo`, `episodes`, `gamma`, `lr`, `hidden_size` |
| **Metrics (every 50 eps)** | `mean_reward`, `portfolio_value` |
| **Final metrics** | `final_portfolio_value`, `total_return_pct`, `sharpe_ratio`, `max_drawdown` |
| **Artifacts** | `{algo}.pth` model weights, reward curve PNG |

```bash
mlflow ui --port 5000   # → http://localhost:5000
```

---

## Tests

```bash
# Run all 44 tests
PYTHONPATH=. pytest tests/ -v

# Individual suites
PYTHONPATH=. pytest tests/test_env.py      # 10 tests — Gymnasium environment
PYTHONPATH=. pytest tests/test_pipeline.py # 17 tests — features + schema validation
PYTHONPATH=. pytest tests/test_api.py      # 17 tests — FastAPI endpoints
```

---

## Design Decisions

**Why a fixed 50-day observation window?**
A rolling window gives the agent time-series context without the full history. 50 bars (~2.5 months) captures medium-term trend patterns. Longer windows increase the observation dimension and slow convergence.

**Why all-in buy / full-sell actions?**
Continuous position sizing adds a continuous action space that requires DDPG/SAC-style agents. Discrete all-or-nothing keeps the action space clean and lets value-based agents (DQN/DDQN) apply directly while still producing meaningful trading signals.

**Why Z-score normalisation per split?**
Each train/test split has a different price distribution. Normalising separately prevents leakage and ensures the agent sees similarly-scaled inputs regardless of absolute price levels.

**Why commission in the env, not post-hoc?**
Commission affects portfolio values that flow back as rewards. Applying it inside `StockTradingEnv.step()` means the agent sees the true cost of each trade during training, producing more realistic learned policies.

**Why policy gradient agents (A2C/PPO) need more episodes?**
Policy gradient methods have high-variance gradient estimates and converge slowly from random initialisation on sparse reward signals. The reward is only non-zero on `Sell` actions; the agent must explore enough buy→hold→sell cycles to get a learning signal. 3000 episodes with the improved reward (holding bonus + scaled profit) gives these agents enough signal to escape "always Hold."

**Why lightweight-charts over Recharts?**
`lightweight-charts` is the same rendering library used by TradingView. It handles 1000+ candle bars in a hardware-accelerated canvas with native crosshairs, zoom, pan, and professional marker rendering — things that Recharts SVG cannot do performantly.

---

## Generalisation

To trade a new asset:

1. **Add data**: `python -m pipeline.ingest --ticker TSLA`
2. **Retrain** (optional, the existing agents will use whatever ticker data is loaded into the env):  
   `python -m src.train --algo all --ticker TSLA --episodes 3000`
3. **Select in UI**: The topbar ticker pills are driven by `/api/tickers` — once `data/processed/TSLA_features.csv` exists, TSLA appears automatically.

The observation space is price-agnostic (Z-scored) so a model trained on AAPL can generalise to other equities, though transfer quality varies with correlation.

---

## Authors

Shubham Sharma (ss695) · tmr32 · jwaxman — Cornell University

*Extended from the original ss695_tmr32_jwaxman final project notebook.*
