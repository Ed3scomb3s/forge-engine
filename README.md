# Forge Engine -- Rust-Accelerated Crypto Futures Backtester

**Code appendix for:** *Algorithmic Alpha in Crypto Futures: A Controlled Empirical Comparison of Reinforcement Learning and Human-Developed Control Strategies*
**Author:** Etienne Descombes | **Supervisor:** Lorenzo Javier Martin Lopez | IE University, 2026

Licensed under the MIT License. See [LICENSE](LICENSE).

## Overview

- Deterministic, candle-based futures backtester with cross/isolated margin, fees, funding rates, TP/SL/liquidations, and slippage modeling.
- Performance-critical engine compiled to native code via a Rust extension (`_rust_core`) using PyO3 and Maturin. The entire state machine, indicators, and RL step loop run at native speed.
- RL training runs at **175,000 steps/sec** (50x speedup over pure Python).
- Vectorized strategy fast path: optimizer backtests run at **~119/sec** (~8ms each) via `VectorStrategy` + Rust-native signal processing.

## Repository Structure

```
forge_engine/            Core Python package
  engine.py              Session management, candle stepping, order execution
  trading.py             Order book, position tracking, P&L, margin, liquidation
  indicators.py          SMA, EMA, RSI, ATR, Bollinger Bands, MACD
  metrics.py             Sharpe, Sortino, max drawdown, Calmar, win rate, etc.
  strategy.py            Base Strategy and VectorStrategy classes
  optuna_optimizer.py    Walk-forward analysis with Optuna HPO
  indexer.py             CSV indexing for O(1) candle lookup
  rl/                    Reinforcement learning module
    env.py               Gymnasium environment (delegates to Rust fast path)
    actions.py           Discrete/Continuous action space definitions
    observations.py      Observation feature specs (OHLCV, indicators, position)
    rewards.py           Reward functions (PnL, Sharpe, DifferentialSharpe, Sortino)
    train.py             SB3 training pipeline (PPO, DQN, A2C)
    agent_strategy.py    Deploy trained RL agents as backtestable strategies
src/                     Rust core (PyO3 extension)
  engine.rs              Full order execution state machine (~2700 lines)
  types.rs               128-bit banker's rounding for deterministic numerics
  indicators.rs          Vectorized indicator computation
  vectorized.rs          Bulk signal processing for fast optimizer path
  rl/                    Rust RL step loop (observation/reward translation)
examples/                Human-designed baseline strategies
  sma_cross.py           Simple Moving Average Crossover
  rsi_reversal.py        RSI Mean-Reversion
  bb_reversion.py        Bollinger Band Mean-Reversion
  macd_momentum.py       MACD Momentum with ATR filter
  buy_and_hold.py        Passive benchmark
examples_rl/             RL experiment configurations
  btc_ppo_v6/            Final thesis experiment (discrete PPO, MACD filter)
  btc_ppo/               Earlier continuous PPO variant
  btc_dqn/               DQN variant
  zec_ppo/               Zcash cross-market robustness test
evaluation/              Thesis evaluation framework
  optimize_baselines.py  Headless Optuna HPO for human baselines
  train_rl_v6.py         V6 training pipeline (5 WFA folds x 3 seeds)
  compare.py             Unified comparison tables
  statistical_tests.py   Bootstrap Sharpe difference tests
  passive_benchmarks.py  Buy-and-hold market benchmarks
  results/               Pre-computed thesis results (JSON)
tests/                   Test suite
```

## Setup

```bash
# Use Python 3.13 for the current PyO3 build
uv venv --python 3.13
source .venv/bin/activate

# Install dependencies and developer tooling
uv sync --group dev

# Build the Rust extension in-place
maturin develop --release --skip-install
```

`stable-baselines3[extra]` is intentionally not used here because it pulls in optional Atari dependencies that require extra system packages and are unrelated to this project.

## Data

The repository stores packaged market data archives in `data/`, but those files are tracked with Git LFS. Clone the repository directly if you want the packaged datasets:

```bash
git clone https://github.com/Ed3scomb3s/forge-engine.git
cd forge-engine
git lfs install
git lfs pull
```

GitHub source ZIP downloads only contain LFS pointer files, not the actual dataset payloads.

You can either use the packaged `data/*.zip` archives or place raw Binance perpetual futures 1-minute OHLCV CSV files in `data/`:
- `BTCUSDT_PERP_1m.csv`
- `ZECUSDT_PERP_1m.csv`

Required columns: `open_time,open,high,low,close,volume,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume`

Funding rate files (optional): `BTCUSDT_PERP_funding.csv`, `ZECUSDT_PERP_funding.csv`

When the packaged zip archives are present, Forge Engine lazily extracts the required CSVs on first use.

## Usage

```bash
# Run a strategy backtest
uv run python examples/sma_cross.py

# Run tests
uv run pytest tests/ -v

# Train the final thesis RL agent
uv run python evaluation/train_rl_v6.py --quick

# Compare all strategies (requires result files)
uv run python -m evaluation.compare

# Run statistical significance tests
uv run python -m evaluation.statistical_tests
```

## Creating a Strategy

**VectorStrategy** (fast, recommended for optimizer):

```python
from forge_engine import VectorStrategy, SMA, RSI

class MyStrategy(VectorStrategy):
    def indicators(self):
        return [SMA(20, source="close", label="SMA20"), RSI(14, source="close", label="RSI14")]

    def signals(self, close, indicators):
        sma = indicators["SMA20"]
        return np.where(close > sma, 1, 0).astype(np.int8)  # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE

    def signal_params(self):
        return {"margin_pct": 0.1, "sl_pct": 0.02, "tp_pct": 0.05}
```

## Notes

- Orders created at candle A are first eligible to fill at A+1.
- Indicators are computed per candle after trading state is updated.
- Set `close_at_end=True` when creating a session to auto-close positions on the final candle.
- `PPO_v6` now consumes explicit MACD line, signal, and histogram observations during training and evaluation.
- If you modify Rust source (`src/`), rebuild with: `maturin develop --release --skip-install`
