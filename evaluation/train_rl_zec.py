"""RL Training + WFA Evaluation for ZEC (Zcash)
================================================
Same protocol as train_rl.py but with ZEC-specific parameters:
- Higher slippage (0.15% vs 0.05%)
- Lower leverage (5x max vs 50x)
- ZECUSDT_PERP symbol

Usage:
    uv run python evaluation/train_rl_zec.py --algo PPO
    uv run python evaluation/train_rl_zec.py --algo DQN
    uv run python evaluation/train_rl_zec.py --algo ALL
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

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor


# ═══════════════════════════════════════════════════════════════════════
# ZEC-specific constants
# ═══════════════════════════════════════════════════════════════════════

SYMBOL = "ZECUSDT_PERP"
TIMEFRAME = "1h"
TF_MINUTES = 60
WARMUP_CANDLES = 50
STARTING_CASH = 100_000
MARGIN_MODE = "cross"
SLIPPAGE_PCT = 0.0015      # 0.15% — 3x higher than BTC (illiquid market)
CLOSE_AT_END = True

FULL_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
FULL_END = datetime(2026, 2, 12, tzinfo=timezone.utc)
N_SPLITS = 5
TEST_RATIO = 0.2
HOLDOUT_RATIO = 0.15
WFA_MODE = "anchored"

SEEDS = [42, 123, 456]
PPO_TRAIN_STEPS = 500_000
DQN_TRAIN_STEPS = 500_000

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SB3_DEVICE = os.getenv("FORGE_SB3_DEVICE", "cpu")


# ═══════════════════════════════════════════════════════════════════════
# PPO Config (adapted for ZEC: lower leverage, wider SL/TP)
# ═══════════════════════════════════════════════════════════════════════

PPO_CONFIG = dict(
    observations=[
        "ohlcv", "returns", "rsi_14", "sma_ratio_20_50",
        "position_info", "drawdown",
    ],
    actions_factory=lambda: ContinuousActions(
        max_margin_pct=0.10, threshold=0.23, sl_pct=0.08, tp_pct=0.16,
    ),
    reward="sortino",
    reward_kwargs=dict(eta=0.021, min_std=0.005),
    engine_kwargs=dict(
        starting_cash=STARTING_CASH, leverage=5, margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT, close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME, warmup_candles=WARMUP_CANDLES, max_steps=720,
    ),
    train_steps=PPO_TRAIN_STEPS,
    algo_factory=lambda env, seed: PPO(
        "MlpPolicy", env,
        learning_rate=9e-5, n_steps=2048, batch_size=64, n_epochs=10,
        ent_coef=0.008, gamma=0.99, gae_lambda=0.953, clip_range=0.30,
        seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    ),
    algo_name="PPO",
)


# ═══════════════════════════════════════════════════════════════════════
# DQN Config (adapted for ZEC)
# ═══════════════════════════════════════════════════════════════════════

DQN_CONFIG = dict(
    observations=["ohlcv", "rsi_14", "atr_14", "position_info", "drawdown"],
    actions_factory=lambda: DiscreteActions(
        actions=["hold", "open_long", "open_short", "close"],
        margin_pct=0.15, sl_pct=0.05, tp_pct=0.10,
    ),
    reward="differential_sharpe",
    reward_kwargs={},
    engine_kwargs=dict(
        starting_cash=STARTING_CASH, leverage=5, margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT, close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME, warmup_candles=WARMUP_CANDLES, max_steps=0,
    ),
    train_steps=DQN_TRAIN_STEPS,
    algo_factory=lambda env, seed: DQN(
        "MlpPolicy", env,
        learning_rate=1e-4, buffer_size=100_000, learning_starts=10_000,
        exploration_fraction=0.3, exploration_final_eps=0.05, batch_size=128,
        seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
    algo_name="DQN",
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
        print("Loading ZEC candle data (one-time)...")

        tmp = create_session(
            symbol=SYMBOL,
            start_date="2022-01-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z",
            starting_cash=STARTING_CASH,
            leverage=5,
            margin_mode=MARGIN_MODE,
            slippage_pct=SLIPPAGE_PCT,
            close_at_end=CLOSE_AT_END,
            timeframe=TIMEFRAME,
            warmup_candles=WARMUP_CANDLES,
            enable_visual=False,
        )
        cls.funding_data = getattr(tmp, "funding_data", None)
        cls.data_dir = tmp.data_dir

        cls.candle_data = preload_candle_data_aggregated(
            symbol=SYMBOL,
            start_date="2022-01-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z",
            base_timeframe="1m",
            target_timeframe_minutes=TF_MINUTES,
            data_dir=cls.data_dir,
            warmup_candles=WARMUP_CANDLES,
        )
        cls._loaded = True
        n = len(cls.candle_data.timestamps_unix) if cls.candle_data else 0
        print(f"  Loaded {n:,} candles in {time.time() - t0:.1f}s\n")


# ═══════════════════════════════════════════════════════════════════════
# Environment factory
# ═══════════════════════════════════════════════════════════════════════

def make_env(config, start_iso, end_iso, max_steps_override=None):
    DataCache.load()
    reward_obj = resolve_reward(config["reward"], **config["reward_kwargs"])
    ek = config["engine_kwargs"].copy()
    if max_steps_override is not None:
        ek["max_steps"] = max_steps_override

    env = ForgeEnv(
        symbol=SYMBOL,
        start_date=start_iso,
        end_date=end_iso,
        observations=config["observations"],
        actions=config["actions_factory"](),
        reward=reward_obj,
        **ek,
    )
    env._candle_data = DataCache.candle_data
    env._funding_data = DataCache.funding_data
    env._resolved_data_dir = DataCache.data_dir
    env._intra_tuples = None
    return Monitor(env)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_on_period(model, config, start_iso, end_iso, seed=42):
    env = make_env(config, start_iso, end_iso, max_steps_override=0)
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


# ═══════════════════════════════════════════════════════════════════════
# WFA Training Loop (same as train_rl.py)
# ═══════════════════════════════════════════════════════════════════════

def dt_to_iso(dt):
    return _isoformat_z(dt)

def dt_to_datestr(dt):
    return dt.strftime("%Y-%m-%d")

def run_wfa_training(config, seeds=SEEDS):
    algo_name = config["algo_name"]
    train_steps = config["train_steps"]

    print("=" * 70)
    print(f"  {algo_name} WFA Training -- {SYMBOL}")
    print("=" * 70)

    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=N_SPLITS, test_ratio=TEST_RATIO, mode=WFA_MODE,
    ))
    splits, holdout = splitter.generate_splits(
        full_start=FULL_START, full_end=FULL_END,
        holdout_ratio=HOLDOUT_RATIO, timeframe_minutes=TF_MINUTES,
    )

    print(f"  Folds: {len(splits)}, Seeds: {len(seeds)}, Steps/fold: {train_steps:,}")
    if holdout:
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
    print()

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        print(f"  Fold {i+1}: Train {dt_to_datestr(tr_s)} -> {dt_to_datestr(tr_e)} | "
              f"Test {dt_to_datestr(te_s)} -> {dt_to_datestr(te_e)}")
    print()

    DataCache.load()

    results = {
        "algorithm": algo_name,
        "symbol": SYMBOL,
        "config": {
            "observations": config["observations"],
            "reward": config["reward"],
            "reward_kwargs": config["reward_kwargs"],
            "train_steps": train_steps,
            "n_splits": N_SPLITS,
            "test_ratio": TEST_RATIO,
            "holdout_ratio": HOLDOUT_RATIO,
            "wfa_mode": WFA_MODE,
            "seeds": seeds,
            "engine_kwargs": {k: v for k, v in config["engine_kwargs"].items()},
        },
        "folds": [],
        "holdout": None,
        "aggregate_oos": {},
    }

    all_test_metrics = []
    t_total_start = time.time()

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
        fold_num = fold_idx + 1
        tr_start_iso = dt_to_iso(tr_start)
        tr_end_iso = dt_to_iso(tr_end)
        te_start_iso = dt_to_iso(te_start)
        te_end_iso = dt_to_iso(te_end)

        print("-" * 70)
        print(f"  Fold {fold_num}/{len(splits)}: "
              f"Train {dt_to_datestr(tr_start)} -> {dt_to_datestr(tr_end)} | "
              f"Test {dt_to_datestr(te_start)} -> {dt_to_datestr(te_end)}")
        print("-" * 70)

        fold_result = {
            "fold": fold_num,
            "train_period": [dt_to_datestr(tr_start), dt_to_datestr(tr_end)],
            "test_period": [dt_to_datestr(te_start), dt_to_datestr(te_end)],
            "seeds": {},
            "mean": {},
            "std": {},
        }

        seed_metrics_list = []

        for seed_idx, seed in enumerate(seeds):
            t_seed_start = time.time()
            print(f"    Seed {seed_idx+1}/{len(seeds)} (seed={seed})... ", end="", flush=True)

            try:
                env = make_env(config, tr_start_iso, tr_end_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                env.close()
                train_elapsed = time.time() - t_seed_start

                metrics = evaluate_on_period(
                    model, config, te_start_iso, te_end_iso, seed=seed,
                )
                eval_elapsed = time.time() - t_seed_start - train_elapsed

                fold_result["seeds"][str(seed)] = metrics
                seed_metrics_list.append(metrics)
                all_test_metrics.append(metrics)

                print(f"Sharpe={metrics['sharpe']:+.3f}  "
                      f"PnL={metrics['pnl_pct']:+.1f}%  "
                      f"DD={metrics['max_dd_pct']:.1f}%  "
                      f"({train_elapsed:.0f}s train, {eval_elapsed:.0f}s eval)")

                del model

            except Exception as e:
                print(f"FAILED: {e}")
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
        ho_start_iso = dt_to_iso(holdout[0])
        ho_end_iso = dt_to_iso(holdout[1])

        print("=" * 70)
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
        print("=" * 70)

        holdout_result = {
            "period": [dt_to_datestr(holdout[0]), dt_to_datestr(holdout[1])],
            "seeds": {},
            "mean": {},
            "std": {},
        }

        ho_train_start_iso = dt_to_iso(FULL_START)
        ho_train_end_iso = ho_start_iso

        ho_metrics_list = []
        for seed_idx, seed in enumerate(seeds):
            t_seed_start = time.time()
            print(f"    Seed {seed_idx+1}/{len(seeds)} (seed={seed})... ", end="", flush=True)

            try:
                env = make_env(config, ho_train_start_iso, ho_train_end_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                env.close()
                train_elapsed = time.time() - t_seed_start

                metrics = evaluate_on_period(
                    model, config, ho_start_iso, ho_end_iso, seed=seed,
                )

                holdout_result["seeds"][str(seed)] = metrics
                ho_metrics_list.append(metrics)

                print(f"Sharpe={metrics['sharpe']:+.3f}  "
                      f"PnL={metrics['pnl_pct']:+.1f}%  "
                      f"DD={metrics['max_dd_pct']:.1f}%  "
                      f"({train_elapsed:.0f}s train)")

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
                  f"(+/-{holdout_result['std'].get('sharpe', 0):.3f})  "
                  f"PnL={holdout_result['mean'].get('pnl_pct', 0):+.1f}%")

        results["holdout"] = holdout_result

    # Aggregate OOS
    if all_test_metrics:
        agg = {}
        for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
            vals = [m[key] for m in all_test_metrics if key in m]
            if vals:
                agg[f"mean_{key}"] = round(float(np.mean(vals)), 4)
                agg[f"std_{key}"] = round(float(np.std(vals)), 4)
                agg[f"median_{key}"] = round(float(np.median(vals)), 4)
        results["aggregate_oos"] = agg

    total_elapsed = time.time() - t_total_start

    print()
    print("=" * 70)
    print(f"  {algo_name} ZEC COMPLETE -- {total_elapsed/60:.1f} min total")
    print("=" * 70)
    if results["aggregate_oos"]:
        agg = results["aggregate_oos"]
        print(f"  Aggregate OOS: Sharpe={agg.get('mean_sharpe', 0):+.3f} "
              f"(+/-{agg.get('std_sharpe', 0):.3f})  "
              f"PnL={agg.get('mean_pnl_pct', 0):+.1f}% "
              f"(+/-{agg.get('std_pnl_pct', 0):.1f}%)")
    if results["holdout"] and results["holdout"].get("mean"):
        ho = results["holdout"]["mean"]
        print(f"  Holdout: Sharpe={ho.get('sharpe', 0):+.3f}  "
              f"PnL={ho.get('pnl_pct', 0):+.1f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rl_zec_{algo_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ZEC RL WFA Training for Thesis")
    parser.add_argument("--algo", choices=["PPO", "DQN", "ALL"], default="ALL",
                        help="Algorithm to train (default: ALL)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds (default: 3)")
    args = parser.parse_args()

    seeds = SEEDS[:args.seeds]

    if args.algo in ("PPO", "ALL"):
        run_wfa_training(PPO_CONFIG, seeds=seeds)
        print()

    if args.algo in ("DQN", "ALL"):
        run_wfa_training(DQN_CONFIG, seeds=seeds)
        print()


if __name__ == "__main__":
    main()
