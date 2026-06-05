# RiskAssessRL — common workflows.
# Assumes an activated virtualenv (see README). Run `make help` for the list.

PY ?= python
PYTHONPATH := .
export PYTHONPATH

.PHONY: help install data train evaluate walkforward generalization \
        test lint backend frontend docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python deps + frontend deps
	pip install -r requirements.txt ruff
	npm --prefix frontend install

data:  ## Download + feature-engineer all tickers (AAPL MSFT GOOGL NVDA SPY)
	$(PY) -m pipeline.ingest

train:  ## Train all agents (DQN/DDQN 2000 eps, A2C/PPO 3000 eps) in parallel
	$(PY) -m src.train --algo DQN  --episodes 2000 & \
	$(PY) -m src.train --algo DDQN --episodes 2000 & \
	$(PY) -m src.train --algo A2C  --episodes 3000 & \
	$(PY) -m src.train --algo PPO  --episodes 3000 & \
	wait

evaluate:  ## Regenerate results/comparison.json on the AAPL test split
	$(PY) -m src.evaluate --mode comparison

walkforward:  ## Run the 5-fold walk-forward backtest -> results/walkforward.json
	$(PY) -m src.evaluate --mode walkforward

generalization:  ## Zero-shot the AAPL DDQN on other tickers -> results/generalization.json
	$(PY) -m src.evaluate --mode generalization

test:  ## Run the test suite
	$(PY) -m pytest tests/ -v

lint:  ## Lint src/ and backend/ (matches CI)
	ruff check src/ backend/

backend:  ## Run the FastAPI backend on :8000
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

frontend:  ## Run the Vite dev server on :3000
	npm --prefix frontend run dev

docker-up:  ## Build + start all services via docker-compose
	docker-compose up --build

docker-down:  ## Stop docker-compose services
	docker-compose down

clean:  ## Remove caches and build artifacts (keeps data/ and models/)
	rm -rf .pytest_cache .ruff_cache **/__pycache__ frontend/dist
