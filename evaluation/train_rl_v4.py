"""RL Training V4 — "The Ultimate Trader"
==========================================
Fundamental rethinking, not just hyperparameter tuning.

Key changes from V1-V3:
1. Training from 2020 (not 2022) — 2 extra years of diverse data (COVID crash,
   2021 bull run, 2021 crash, 2022 bear). Agent sees bull, bear, sideways, crash.
2. max_steps=720 (30-day episodes) — V3 used full episodes (1 ep per fold).
   Short episodes = thousands of episodes per training = much more diverse experience.
3. Differential Sharpe reward — tracks Sharpe online, naturally punishes
   overtrading (each unnecessary trade hurts the ratio).
4. Small network [64, 32] — FORCE simplicity. Complex networks memorize,
   simple networks generalize. The MACD strategy that beats us is 5 lines of code.
5. Higher leverage (3x) — amplify whatever signal the agent finds.
6. Observation normalization via VecNormalize.
7. Also try SAC — more sample efficient than PPO for continuous control.

Usage:
    uv run python evaluation/train_rl_v4.py --algo PPO
    uv run python evaluation/train_rl_v4.py --algo SAC
    uv run python evaluation/train_rl_v4.py --algo ALL
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge_engine.rl.env import ForgeEnv
from forge_engine.rl.actions import ContinuousActions, DiscreteActions
from forge_engine.rl.rewards import resolve_reward
from forge_engine.engine import (
    create_session,
    preload_candle_data_aggregated,
    _isoformat_z,
)
from forge_engine.optuna_optimizer import WalkForwardSplitter, WalkForwardConfig
from evaluation.artifacts import evaluate_rl_model_on_period

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

SYMBOL = "BTCUSDT_PERP"
TIMEFRAME = "1h"
TF_MINUTES = 60
WARMUP_CANDLES = 50
STARTING_CASH = 100_000
MARGIN_MODE = "cross"
SLIPPAGE_PCT = 0.0005
CLOSE_AT_END = True

# V4: Start from 2020 — 2 extra years of diverse market data
FULL_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
FULL_END = datetime(2026, 2, 12, tzinfo=timezone.utc)
N_SPLITS = 5
TEST_RATIO = 0.2
HOLDOUT_RATIO = 0.15
WFA_MODE = "anchored"

SEEDS = [42, 123, 456]
PPO_TRAIN_STEPS = 3_000_000     # More steps to learn from more data
SAC_TRAIN_STEPS = 1_000_000     # SAC is more sample efficient

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SB3_DEVICE = os.getenv("FORGE_SB3_DEVICE", "cpu")


# ═══════════════════════════════════════════════════════════════════════
# V4 PPO Config — "Simple but smart"
# ═══════════════════════════════════════════════════════════════════════

PPO_V4_CONFIG = dict(
    observations=[
        "ohlcv",
        "returns",
        "rsi_14",
        "atr_14",
        "sma_ratio_20_50",
        "position_info",
        "drawdown",
    ],
    actions_factory=lambda: ContinuousActions(
        max_margin_pct=0.15,     # Larger positions when confident
        threshold=0.20,          # Slightly tighter dead zone — trade when you see something
        sl_pct=0.03,             # Tight SL — cut losers fast
        tp_pct=0.09,             # 3:1 ratio
    ),
    reward="differential_sharpe",  # V4: online Sharpe — penalizes overtrading naturally
    reward_kwargs=dict(
        eta=0.01,
        scale=1.0,
    ),
    engine_kwargs=dict(
        starting_cash=STARTING_CASH,
        leverage=3,              # V4: higher leverage to amplify signal
        margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT,
        close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME,
        warmup_candles=WARMUP_CANDLES,
        max_steps=720,           # V4: 30-day episodes for diversity
    ),
    train_steps=PPO_TRAIN_STEPS,
    algo_factory=lambda env, seed: PPO(
        "MlpPolicy", env,
        learning_rate=1e-4,      # Standard LR
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.015,          # Moderate entropy
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 32], vf=[64, 32]),  # SMALL — forces generalization
        ),
    ),
    algo_name="PPO_v4",
    use_vecnorm=True,            # V4: normalize observations and rewards
)


# ═══════════════════════════════════════════════════════════════════════
# V4 SAC Config — Off-policy, more sample efficient
# ═══════════════════════════════════════════════════════════════════════

SAC_V4_CONFIG = dict(
    observations=[
        "ohlcv",
        "returns",
        "rsi_14",
        "atr_14",
        "sma_ratio_20_50",
        "position_info",
        "drawdown",
    ],
    actions_factory=lambda: ContinuousActions(
        max_margin_pct=0.15,
        threshold=0.20,
        sl_pct=0.03,
        tp_pct=0.09,
    ),
    reward="differential_sharpe",
    reward_kwargs=dict(eta=0.01, scale=1.0),
    engine_kwargs=dict(
        starting_cash=STARTING_CASH,
        leverage=3,
        margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT,
        close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME,
        warmup_candles=WARMUP_CANDLES,
        max_steps=720,
    ),
    train_steps=SAC_TRAIN_STEPS,
    algo_factory=lambda env, seed: SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",         # SAC auto-tunes entropy — key advantage
        seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 32], qf=[64, 32]),
        ),
    ),
    algo_name="SAC_v4",
    use_vecnorm=True,
)


# ═══════════════════════════════════════════════════════════════════════
# Data Cache
# ═══════════════════════════════════════════════════════════════════════

class DataCache:
    candle_data = None
    funding_data = None
    data_dir = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        t0 = time.time()
        print("Loading candle data from 2020 (one-time)...")
        tmp = create_session(
            symbol=SYMBOL, start_date="2020-01-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z", starting_cash=STARTING_CASH,
            leverage=3, margin_mode=MARGIN_MODE, slippage_pct=SLIPPAGE_PCT,
            close_at_end=CLOSE_AT_END, timeframe=TIMEFRAME,
            warmup_candles=WARMUP_CANDLES, enable_visual=False,
        )
        cls.funding_data = getattr(tmp, "funding_data", None)
        cls.data_dir = tmp.data_dir
        cls.candle_data = preload_candle_data_aggregated(
            symbol=SYMBOL, start_date="2020-01-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z", base_timeframe="1m",
            target_timeframe_minutes=TF_MINUTES, data_dir=cls.data_dir,
            warmup_candles=WARMUP_CANDLES,
        )
        cls._loaded = True
        n = len(cls.candle_data.timestamps_unix) if cls.candle_data else 0
        print(f"  Loaded {n:,} candles in {time.time() - t0:.1f}s\n")


def make_env(config, start_iso, end_iso, max_steps_override=None):
    DataCache.load()
    reward_obj = resolve_reward(config["reward"], **config["reward_kwargs"])
    ek = config["engine_kwargs"].copy()
    if max_steps_override is not None:
        ek["max_steps"] = max_steps_override
    env = ForgeEnv(
        symbol=SYMBOL, start_date=start_iso, end_date=end_iso,
        observations=config["observations"],
        actions=config["actions_factory"](),
        reward=reward_obj, **ek,
    )
    env._candle_data = DataCache.candle_data
    env._funding_data = DataCache.funding_data
    env._resolved_data_dir = DataCache.data_dir
    env._intra_tuples = None
    return env


def make_train_env(config, start_iso, end_iso):
    """Create training env with optional VecNormalize wrapper."""
    def _make():
        return Monitor(make_env(config, start_iso, end_iso))

    if config.get("use_vecnorm", False):
        venv = DummyVecEnv([_make])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0, gamma=0.99)
        return venv
    else:
        return Monitor(make_env(config, start_iso, end_iso))


def evaluate_on_period(model, config, start_iso, end_iso, seed=42):
    """Evaluate without VecNormalize to get true metrics."""
    env = Monitor(make_env(config, start_iso, end_iso, max_steps_override=0))
    try:
        return evaluate_rl_model_on_period(
            model,
            env,
            starting_cash=float(config["engine_kwargs"]["starting_cash"]),
            start_iso=start_iso,
            end_iso=end_iso,
            timeframe=str(config["engine_kwargs"]["timeframe"]),
            seed=seed,
        )
    finally:
        env.close()


def dt_to_iso(dt): return _isoformat_z(dt)
def dt_to_datestr(dt): return dt.strftime("%Y-%m-%d")


def run_wfa_training(config, seeds=SEEDS):
    algo_name = config["algo_name"]
    train_steps = config["train_steps"]

    print("=" * 70)
    print(f"  {algo_name} WFA Training (V4) -- {SYMBOL}")
    print("=" * 70)

    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=N_SPLITS, test_ratio=TEST_RATIO, mode=WFA_MODE,
    ))
    splits, holdout = splitter.generate_splits(
        full_start=FULL_START, full_end=FULL_END,
        holdout_ratio=HOLDOUT_RATIO, timeframe_minutes=TF_MINUTES,
    )

    print(f"  Folds: {len(splits)}, Seeds: {len(seeds)}, Steps/fold: {train_steps:,}")
    print(f"  Data from: {dt_to_datestr(FULL_START)} (includes 2020 crash, 2021 bull/bear)")
    print(f"  VecNormalize: {config.get('use_vecnorm', False)}")
    if holdout:
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
    print()
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        print(f"  Fold {i+1}: Train {dt_to_datestr(tr_s)} -> {dt_to_datestr(tr_e)} | "
              f"Test {dt_to_datestr(te_s)} -> {dt_to_datestr(te_e)}")
    print()

    DataCache.load()

    results = {
        "algorithm": algo_name, "symbol": SYMBOL, "version": "v4",
        "config": {
            "observations": config["observations"],
            "reward": config["reward"], "reward_kwargs": config["reward_kwargs"],
            "train_steps": train_steps, "n_splits": N_SPLITS,
            "test_ratio": TEST_RATIO, "holdout_ratio": HOLDOUT_RATIO,
            "wfa_mode": WFA_MODE, "seeds": seeds,
            "engine_kwargs": {k: v for k, v in config["engine_kwargs"].items()},
            "use_vecnorm": config.get("use_vecnorm", False),
        },
        "folds": [], "holdout": None, "aggregate_oos": {},
    }

    all_test_metrics = []
    t_total_start = time.time()

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
        fold_num = fold_idx + 1
        tr_s_iso, tr_e_iso = dt_to_iso(tr_start), dt_to_iso(tr_end)
        te_s_iso, te_e_iso = dt_to_iso(te_start), dt_to_iso(te_end)

        print("-" * 70)
        print(f"  Fold {fold_num}/{len(splits)}: "
              f"Train {dt_to_datestr(tr_start)} -> {dt_to_datestr(tr_end)} | "
              f"Test {dt_to_datestr(te_start)} -> {dt_to_datestr(te_end)}")
        print("-" * 70)

        fold_result = {
            "fold": fold_num,
            "train_period": [dt_to_datestr(tr_start), dt_to_datestr(tr_end)],
            "test_period": [dt_to_datestr(te_start), dt_to_datestr(te_end)],
            "seeds": {}, "mean": {}, "std": {},
        }
        seed_metrics_list = []

        for seed_idx, seed in enumerate(seeds):
            t0 = time.time()
            print(f"    Seed {seed_idx+1}/{len(seeds)} (seed={seed})... ", end="", flush=True)
            try:
                env = make_train_env(config, tr_s_iso, tr_e_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                if hasattr(env, 'close'):
                    env.close()
                train_t = time.time() - t0
                metrics = evaluate_on_period(model, config, te_s_iso, te_e_iso, seed=seed)
                fold_result["seeds"][str(seed)] = metrics
                seed_metrics_list.append(metrics)
                all_test_metrics.append(metrics)
                print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  "
                      f"DD={metrics['max_dd_pct']:.1f}%  active={metrics['n_active_steps']}/{metrics['total_steps']}  "
                      f"({train_t:.0f}s)")
                del model
            except Exception as e:
                import traceback
                print(f"FAILED: {e}")
                traceback.print_exc()
                fold_result["seeds"][str(seed)] = {"error": str(e)}

        if seed_metrics_list:
            for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
                vals = [m[key] for m in seed_metrics_list if key in m]
                if vals:
                    fold_result["mean"][key] = round(float(np.mean(vals)), 4)
                    fold_result["std"][key] = round(float(np.std(vals)), 4)
            print(f"    Fold {fold_num} mean: Sharpe={fold_result['mean'].get('sharpe', 0):+.3f} "
                  f"(+/-{fold_result['std'].get('sharpe', 0):.3f})  "
                  f"PnL={fold_result['mean'].get('pnl_pct', 0):+.1f}%")
        results["folds"].append(fold_result)
        print()

    # Holdout
    if holdout:
        ho_s_iso, ho_e_iso = dt_to_iso(holdout[0]), dt_to_iso(holdout[1])
        print("=" * 70)
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
        print("=" * 70)
        holdout_result = {
            "period": [dt_to_datestr(holdout[0]), dt_to_datestr(holdout[1])],
            "seeds": {}, "mean": {}, "std": {},
        }
        ho_metrics_list = []
        for seed_idx, seed in enumerate(seeds):
            t0 = time.time()
            print(f"    Seed {seed_idx+1}/{len(seeds)} (seed={seed})... ", end="", flush=True)
            try:
                env = make_train_env(config, dt_to_iso(FULL_START), ho_s_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                if hasattr(env, 'close'):
                    env.close()
                train_t = time.time() - t0
                metrics = evaluate_on_period(model, config, ho_s_iso, ho_e_iso, seed=seed)
                holdout_result["seeds"][str(seed)] = metrics
                ho_metrics_list.append(metrics)
                print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  "
                      f"DD={metrics['max_dd_pct']:.1f}%  ({train_t:.0f}s)")
                del model
            except Exception as e:
                print(f"FAILED: {e}")
                holdout_result["seeds"][str(seed)] = {"error": str(e)}
        if ho_metrics_list:
            for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
                vals = [m[key] for m in ho_metrics_list if key in m]
                if vals:
                    holdout_result["mean"][key] = round(float(np.mean(vals)), 4)
                    holdout_result["std"][key] = round(float(np.std(vals)), 4)
            print(f"    Holdout mean: Sharpe={holdout_result['mean'].get('sharpe', 0):+.3f} "
                  f"PnL={holdout_result['mean'].get('pnl_pct', 0):+.1f}%")
        results["holdout"] = holdout_result

    # Aggregate
    if all_test_metrics:
        agg = {}
        for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
            vals = [m[key] for m in all_test_metrics if key in m]
            if vals:
                agg[f"mean_{key}"] = round(float(np.mean(vals)), 4)
                agg[f"std_{key}"] = round(float(np.std(vals)), 4)
        results["aggregate_oos"] = agg

    total_elapsed = time.time() - t_total_start
    print()
    print("=" * 70)
    print(f"  {algo_name} COMPLETE -- {total_elapsed/60:.1f} min total")
    print("=" * 70)
    if results["aggregate_oos"]:
        agg = results["aggregate_oos"]
        print(f"  Aggregate OOS: Sharpe={agg.get('mean_sharpe', 0):+.3f} "
              f"(+/-{agg.get('std_sharpe', 0):.3f})  "
              f"PnL={agg.get('mean_pnl_pct', 0):+.1f}%")
    if results["holdout"] and results["holdout"].get("mean"):
        ho = results["holdout"]["mean"]
        print(f"  Holdout: Sharpe={ho.get('sharpe', 0):+.3f}  PnL={ho.get('pnl_pct', 0):+.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rl_btc_{algo_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RL V4 WFA Training — Ultimate Trader")
    parser.add_argument("--algo", choices=["PPO", "SAC", "ALL"], default="ALL")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    seeds = SEEDS[:args.seeds]
    if args.algo in ("PPO", "ALL"):
        run_wfa_training(PPO_V4_CONFIG, seeds=seeds)
        print()
    if args.algo in ("SAC", "ALL"):
        run_wfa_training(SAC_V4_CONFIG, seeds=seeds)
        print()


if __name__ == "__main__":
    main()
