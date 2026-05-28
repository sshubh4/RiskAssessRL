# RiskAssessRL

> Reinforcement learning platform for risk-adjusted trading strategy simulation — with a TradingView-style live dashboard.

[![Dashboard](https://img.shields.io/badge/Dashboard-localhost%3A3000-2962ff?style=flat-square)](http://localhost:3000)
[![MLflow](https://img.shields.io/badge/MLflow_UI-localhost%3A5000-0194e2?style=flat-square)](http://localhost:5000)
[![API Docs](https://img.shields.io/badge/API_Docs-localhost%3A8000%2Fdocs-009688?style=flat-square)](http://localhost:8000/docs)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-44%20passing-26a69a?style=flat-square)](tests/)

---

## What It Does

Five reinforcement learning agents (Random, DQN, DDQN, A2C, PPO) trade a configurable stock portfolio against historical OHLCV data. A live dashboard streams trades step-by-step over WebSocket, renders them on a custom SVG candlestick chart (with Bollinger Band overlays and trade markers), plots portfolio equity vs buy-and-hold, and displays Sharpe ratio, max drawdown, win rate, and alpha — in real time.

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
│  │  hand-built SVG  │  │  REST + WS       │  │  mlruns/    │  │
│  │  candlestick &   │  │  RL agents       │  │             │  │
│  │  portfolio charts│  │  SL/TP/DD logic  │  │             │  │
│  └────────┬─────────┘  └────────┬─────────┘  └─────────────┘  │
│           │   WebSocket /ws/simulate           │               │
│           └───────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴──────────────────┐
          │             pipeline/                 │
          │  ingest → features → validate         │
          │  • 5 tickers: AAPL MSFT GOOGL NVDA SPY│
          │  • Parallel ThreadPoolExecutor        │
          │  • APScheduler: daily 18:00 ET (AAPL) │
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

# 4. Train all agents (default episode counts; increase for better A2C/PPO convergence)
PYTHONPATH=. python -m src.train --algo DQN  --episodes 2000 &
PYTHONPATH=. python -m src.train --algo DDQN --episodes 2000 &
PYTHONPATH=. python -m src.train --algo A2C  --episodes 3000 &
PYTHONPATH=. python -m src.train --algo PPO  --episodes 3000 &
wait

# 5. Evaluate and generate results/comparison.json
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
# After running steps 1–5 above to populate data/ and models/:
docker-compose up --build
```

Services:
- Dashboard: http://localhost:3000
- API + Swagger docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## Project Structure

```
RiskAssessRL/
├── pipeline/
│   ├── ingest.py        # yfinance download → data/raw/; run_all() for 5 tickers in parallel
│   ├── features.py      # MA20/50, RSI-14, MACD, Bollinger Bands, ATR-14, Support/Resistance
│   ├── validate.py      # Pandera schema: nulls, price > 0, RSI in [0,100], monotonic dates
│   └── scheduler.py     # APScheduler BlockingScheduler — runs AAPL pipeline daily at 18:00 ET
├── src/
│   ├── env.py           # StockTradingEnv: 50-day window, 9 features, Discrete(3), commission
│   ├── agents.py        # RandomAgent, DQNAgent, DoubleDQNAgent, A2CAgent, PPOAgent
│   ├── train.py         # MLflow-instrumented training; CLI: python -m src.train --algo X
│   └── evaluate.py      # sharpe_ratio, max_drawdown, total_return_pct, run_comparison()
├── backend/
│   ├── main.py          # FastAPI: REST + WebSocket; SL/TP/max-DD kill switch; multi-ticker
│   └── Dockerfile
├── frontend/
│   ├── Dockerfile
│   └── src/
│       ├── App.jsx                   # Root: WebSocket state, B&H computation, layout
│       └── components/
│           ├── Topbar.jsx            # 48px topbar: ticker pills, price, clock, status
│           ├── LeftPanel.jsx         # Strategy selector, backtest params, risk controls
│           ├── TradingChart.jsx      # Custom SVG candlestick + volume + Bollinger Bands
│           ├── PortfolioChart.jsx    # Custom SVG line chart: portfolio vs B&H
│           ├── ComparisonTable.jsx   # All-algo table with best-value highlighting
│           └── AlgorithmsTab.jsx     # Per-algo cards: architecture, metrics, apply settings
├── tests/
│   ├── test_env.py        # 13 tests: obs shape, capital, actions, step log, split
│   ├── test_pipeline.py   # 14 tests: indicators, schema validation, edge cases
│   └── test_api.py        # 17 tests: health, algorithms, data, simulate endpoints
├── docs/
│   └── archive/
│       └── original_course_project/  # Original course notebook, prototype weights, report
├── data/
│   ├── raw/               # Raw OHLCV CSVs from yfinance
│   └── processed/         # Feature-engineered CSVs (committed as source of truth)
├── models/                # Trained .pth weight files (committed for reproducibility)
├── results/
│   └── comparison.json    # Latest agent evaluation results
├── docker-compose.yml
└── requirements.txt
```

---

## Algorithms

| Algorithm | Type | Replay Buffer | Architecture | Default Training |
|---|---|---|---|---|
| **Random** | Baseline | — | Uniform random over {Buy, Sell, Hold} | None |
| **DQN** | Value-based, off-policy | 10 000 transitions | 3-layer MLP (451→128→64→3), ε-greedy | 2 000 episodes |
| **DDQN** | Value-based, off-policy | 10 000 transitions | Dual MLP (online + target), decoupled argmax | 2 000 episodes |
| **A2C** | Policy gradient, on-policy | — | Shared trunk (451→128→64) → actor + critic heads | 300 episodes |
| **PPO** | Policy gradient, on-policy | — | Actor-critic, clipped ratio ε=0.2, GAE λ=0.95 | 300 episodes |

**Observation**: 50-day window × 9 Z-scored features + 1 position flag = **451-dimensional** vector

**Features**: MA\_20, MA\_50, RSI\_14, MACD, BB\_upper, BB\_lower, ATR\_14, Support, Resistance

**Actions**: `0` = Buy (deploy `position_size × capital`), `1` = Sell all, `2` = Hold

**Reward**: `(portfolio_Δ / initial_capital) × 10` − `std(recent_returns) × 0.05 × risk_aversion` − `0.1` on invalid action

---

## Algorithm Results (AAPL, 20% test split, $100 000 starting capital)

| Algorithm | Return | Sharpe | Max Drawdown | vs AAPL B&H | vs SPY B&H |
|---|---|---|---|---|---|
| Random | +5.48% | 0.280 | −12.1% | −47.2pp | −20.3pp |
| DQN | +12.35% | 0.811 | −11.0% | −40.3pp | −13.4pp |
| **DDQN** | **+29.94%** | **1.928** | **−9.3%** | **−22.8pp** | **+4.2pp** |
| A2C | 0.00% | 0.000 | 0.0% | — | — |
| PPO | 0.00% | 0.000 | 0.0% | — | — |
| SPY B&H | +25.78% | — | — | — | — |
| AAPL B&H | +52.70% | — | — | — | — |

> **Note**: A2C and PPO are trained for only 300 episodes by default, which is insufficient for on-policy methods to escape the hold-only local optimum. Training A2C/PPO for 3 000+ episodes with a higher entropy coefficient yields non-trivial strategies. DDQN already outperforms SPY buy-and-hold by ~4 percentage points on the held-out test set.

---

## Risk Controls (Inference-Time)

All risk parameters are applied as **action overrides at inference time** — they do not require retraining.

| Control | Effect |
|---|---|
| **Stop Loss** | Forces a sell when current price falls X% below entry price |
| **Take Profit** | Forces a sell when current price rises X% above entry price |
| **Max Drawdown Kill Switch** | Halts simulation when portfolio drawdown from peak exceeds threshold |
| **Position Size** | Fraction of available capital deployed on each Buy signal (0.1 – 1.0) |
| **Risk Aversion** | Scales the volatility penalty in the reward signal at inference time |
| **Custom Test Range** | Slider-selectable start/end date window within the available history |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status, loaded model names, available tickers |
| `GET` | `/api/tickers` | List of tickers with available processed data |
| `GET` | `/api/algorithms` | All-algo metrics from `results/comparison.json` |
| `GET` | `/api/data?ticker=AAPL` | OHLCV for all available bars |
| `POST` | `/api/backtest` | Full simulation with metrics; supports SL/TP/DD/date range |
| `POST` | `/api/run_all_backtest` | Runs all 5 algos under identical params; returns comparison + benchmarks |
| `WS` | `/ws/simulate` | Real-time step streaming: speed control, SL/TP/DD, date range |

Interactive docs: **http://localhost:8000/docs**

---

## Pipeline

```bash
# Single ticker
PYTHONPATH=. python -m pipeline.ingest --ticker AAPL

# All 5 tickers in parallel (AAPL MSFT GOOGL NVDA SPY)
PYTHONPATH=. python -m pipeline.ingest

# Start the daily scheduler (18:00 ET, AAPL)
PYTHONPATH=. python -m pipeline.scheduler
```

The scheduler runs AAPL only. To schedule additional tickers, edit `pipeline/scheduler.py`.

---

## MLflow Experiment Tracking

Each `src.train` run automatically logs to MLflow:

| Category | Items |
|---|---|
| **Parameters** | `algo`, `episodes`, `gamma`, `lr`, `hidden_size`, `batch_size` |
| **Metrics (every 50 eps)** | `mean_reward`, `portfolio_value` |
| **Final metrics** | `final_portfolio_value`, `total_return_pct`, `sharpe_ratio`, `max_drawdown` |
| **Artifacts** | `{algo}.pth` weights, reward-curve PNG |

```bash
mlflow ui --port 5000   # → http://localhost:5000
```

---

## Tests

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Individual suites
PYTHONPATH=. pytest tests/test_env.py       # 13 tests — Gymnasium environment
PYTHONPATH=. pytest tests/test_pipeline.py  # 14 tests — features + Pandera schema
PYTHONPATH=. pytest tests/test_api.py       # 17 tests — FastAPI endpoints
```

---

## Design Decisions

**Why a 50-day observation window?**
50 bars (~2.5 months) captures medium-term trend structure. Shorter windows lose trend context; longer windows increase the observation dimension and slow convergence disproportionately.

**Why discrete all-in / all-out actions?**
Continuous position sizing requires actor-critic methods designed for continuous action spaces (DDPG, SAC). Discrete Buy/Sell/Hold keeps the action space minimal, lets DQN/DDQN apply directly, and still produces interpretable trading signals.

**Why Z-score normalisation per split?**
Normalising each split separately prevents data leakage (no test-period statistics leak into training) and ensures the agent sees consistently-scaled inputs regardless of absolute price level.

**Why action-override risk controls rather than reward shaping?**
Stop-loss/take-profit as reward penalties would require retraining every time a user adjusts thresholds. Applying them as inference-time action overrides inside `_run_simulation` means any threshold is available instantly without touching the trained weights.

**Why hand-built SVG charts?**
The charts required precise control over candle geometry, trade-marker overlays, and ResizeObserver-driven re-layout that charting library abstractions made harder, not easier. Custom SVG is ~300 lines and has zero runtime dependencies.

**Why are A2C and PPO 0% return by default?**
On-policy methods have high-variance gradient estimates and converge slowly from random initialisation. The default 300 training episodes is a quick smoke-test value. Train A2C/PPO for 3 000+ episodes to see real strategies emerge.

---

## Authors

Shubham Sharma · Cornell University

*Extended from the original course project (see `docs/archive/original_course_project/`).*
