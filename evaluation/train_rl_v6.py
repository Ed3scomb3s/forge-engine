"""RL Training V6 — "The MACD Filter"
======================================
Completely different paradigm: instead of learning trading from scratch,
the agent learns WHEN to follow a known-profitable signal (MACD crossover).

V1-V5 failed because continuous actions let the agent always trade.
V6 uses DISCRETE actions (hold/long/short/close) with the MACD crossover
signal directly in the observation space. The agent's job is NOT to predict
price direction — it's to learn which MACD signals to trust and which to ignore.

Key insight: MACD Momentum baseline makes 5 trades with 80% win rate.
If the RL agent can learn to be even more selective, it wins.

Usage:
    uv run python evaluation/train_rl_v6.py --quick
    uv run python evaluation/train_rl_v6.py
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
from forge_engine.rl.actions import DiscreteActions
from forge_engine.rl.rewards import resolve_reward
from forge_engine.engine import (
    create_session, preload_candle_data_aggregated, _isoformat_z,
)
from forge_engine.optuna_optimizer import WalkForwardSplitter, WalkForwardConfig
from evaluation.artifacts import build_model_eval_env, evaluate_rl_model_on_period

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SYMBOL = "BTCUSDT_PERP"
TIMEFRAME = "1h"
TF_MINUTES = 60
WARMUP_CANDLES = 50
STARTING_CASH = 100_000
MARGIN_MODE = "cross"
SLIPPAGE_PCT = 0.0005
CLOSE_AT_END = True

FULL_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
FULL_END = datetime(2026, 2, 12, tzinfo=timezone.utc)
N_SPLITS = 5
TEST_RATIO = 0.2
HOLDOUT_RATIO = 0.15
WFA_MODE = "anchored"

SEEDS = [42, 123, 456]
TRAIN_STEPS = 1_000_000

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SB3_DEVICE = os.getenv("FORGE_SB3_DEVICE", "cpu")


PPO_V6_CONFIG = dict(
    observations=[
        "ohlcv",
        "returns",
        "rsi_14",
        "atr_14",
        "sma_ratio_20_50",
        "sma_ratio_10_30",
        "macd_line_12_26_9",
        "macd_signal_12_26_9",
        "macd_hist_12_26_9",
        "position_info",
        "drawdown",
    ],
    actions_factory=lambda: DiscreteActions(
        actions=["hold", "open_long", "open_short", "close"],
        margin_pct=0.10,         # Fixed 10% position size
        sl_pct=0.03,             # 3% stop loss
        tp_pct=0.09,             # 9% take profit (3:1 ratio)
    ),
    reward="differential_sharpe",  # Online Sharpe — penalizes overtrading naturally
    reward_kwargs=dict(eta=0.005, scale=1.0),  # Lower eta = longer memory = more stable
    engine_kwargs=dict(
        starting_cash=STARTING_CASH,
        leverage=2,
        margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT,
        close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME,
        warmup_candles=WARMUP_CANDLES,
        max_steps=720,           # 30-day episodes
    ),
    train_steps=TRAIN_STEPS,
    algo_factory=lambda env, seed: PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,      # Standard LR for discrete
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,           # Standard entropy
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
    ),
    algo_name="PPO_v6",
)


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
        print("Loading candle data from 2020...")
        tmp = create_session(
            symbol=SYMBOL, start_date="2020-01-01T00:00:00Z",
            end_date="2026-03-01T00:00:00Z", starting_cash=STARTING_CASH,
            leverage=2, margin_mode=MARGIN_MODE, slippage_pct=SLIPPAGE_PCT,
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


def make_env(start_iso, end_iso, max_steps_override=None):
    DataCache.load()
    config = PPO_V6_CONFIG
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


def make_train_env(start_iso, end_iso):
    def _make():
        return Monitor(make_env(start_iso, end_iso))
    venv = DummyVecEnv([_make])
    return VecNormalize(venv, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99)


def evaluate_on_period(model, start_iso, end_iso, seed=42):
    env = build_model_eval_env(
        lambda: Monitor(make_env(start_iso, end_iso, max_steps_override=0)),
        model,
    )
    try:
        return evaluate_rl_model_on_period(
            model,
            env,
            starting_cash=STARTING_CASH,
            start_iso=start_iso,
            end_iso=end_iso,
            timeframe=TIMEFRAME,
            seed=seed,
        )
    finally:
        env.close()


def dt_to_iso(dt): return _isoformat_z(dt)
def dt_to_datestr(dt): return dt.strftime("%Y-%m-%d")


def run_training(seeds=SEEDS, quick=False, train_steps_override=None):
    config = PPO_V6_CONFIG
    algo_name = config["algo_name"]
    train_steps = 500_000 if quick else int(train_steps_override or config["train_steps"])
    n_splits = 1 if quick else N_SPLITS
    seeds_to_use = seeds[:1] if quick else seeds

    print("=" * 70)
    print(f"  {algo_name} {'QUICK TEST' if quick else 'WFA Training'} -- {SYMBOL}")
    print("=" * 70)

    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=n_splits, test_ratio=TEST_RATIO, mode=WFA_MODE,
    ))
    splits, holdout = splitter.generate_splits(
        full_start=FULL_START, full_end=FULL_END,
        holdout_ratio=HOLDOUT_RATIO, timeframe_minutes=TF_MINUTES,
    )

    print(f"  Folds: {len(splits)}, Seeds: {len(seeds_to_use)}, Steps: {train_steps:,}")
    print(f"  DISCRETE actions: hold/long/short/close")
    print(f"  SL/TP: 3%/9% (3:1 ratio), leverage 2x")
    print(f"  Reward: differential_sharpe (eta=0.005)")
    if holdout:
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
    print()

    DataCache.load()

    results = {
        "algorithm": algo_name, "symbol": SYMBOL, "version": "v6",
        "config": {
            "observations": config["observations"],
            "reward": config["reward"], "reward_kwargs": config["reward_kwargs"],
            "train_steps": train_steps, "n_splits": len(splits),
            "seeds": list(seeds_to_use),
            "engine_kwargs": {k: v for k, v in config["engine_kwargs"].items()},
        },
        "folds": [], "holdout": None, "aggregate_oos": {},
    }

    all_test_metrics = []
    t_total_start = time.time()

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
        fold_num = fold_idx + 1
        tr_s_iso, tr_e_iso = dt_to_iso(tr_start), dt_to_iso(tr_end)
        te_s_iso, te_e_iso = dt_to_iso(te_start), dt_to_iso(te_end)

        print(f"  Fold {fold_num}/{len(splits)}: "
              f"Train {dt_to_datestr(tr_start)} -> {dt_to_datestr(tr_end)} | "
              f"Test {dt_to_datestr(te_start)} -> {dt_to_datestr(te_end)}")

        fold_result = {
            "fold": fold_num,
            "train_period": [dt_to_datestr(tr_start), dt_to_datestr(tr_end)],
            "test_period": [dt_to_datestr(te_start), dt_to_datestr(te_end)],
            "seeds": {}, "mean": {}, "std": {},
        }
        seed_metrics_list = []

        for seed_idx, seed in enumerate(seeds_to_use):
            t0 = time.time()
            print(f"    Seed {seed} ... ", end="", flush=True)
            try:
                env = make_train_env(tr_s_iso, tr_e_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                env.close()
                train_t = time.time() - t0
                metrics = evaluate_on_period(model, te_s_iso, te_e_iso, seed=seed)
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
            print(f"    -> mean Sharpe={fold_result['mean'].get('sharpe', 0):+.3f} "
                  f"PnL={fold_result['mean'].get('pnl_pct', 0):+.1f}%")
        results["folds"].append(fold_result)

        # Checkpoint
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"rl_btc_{algo_name.lower()}_checkpoint.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print()

    # Holdout
    if holdout and not quick:
        ho_s_iso, ho_e_iso = dt_to_iso(holdout[0]), dt_to_iso(holdout[1])
        print(f"  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")
        holdout_result = {
            "period": [dt_to_datestr(holdout[0]), dt_to_datestr(holdout[1])],
            "seeds": {}, "mean": {}, "std": {},
        }
        ho_metrics_list = []
        for seed in seeds_to_use:
            t0 = time.time()
            print(f"    Seed {seed} ... ", end="", flush=True)
            try:
                env = make_train_env(dt_to_iso(FULL_START), ho_s_iso)
                model = config["algo_factory"](env, seed)
                model.learn(total_timesteps=train_steps)
                env.close()
                metrics = evaluate_on_period(model, ho_s_iso, ho_e_iso, seed=seed)
                holdout_result["seeds"][str(seed)] = metrics
                ho_metrics_list.append(metrics)
                print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  "
                      f"DD={metrics['max_dd_pct']:.1f}%  ({time.time()-t0:.0f}s)")
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
            print(f"    -> Holdout mean: Sharpe={holdout_result['mean'].get('sharpe', 0):+.3f} "
                  f"PnL={holdout_result['mean'].get('pnl_pct', 0):+.1f}%")
        results["holdout"] = holdout_result

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
    print(f"  {algo_name} COMPLETE -- {total_elapsed/60:.1f} min")
    print("=" * 70)
    if results["aggregate_oos"]:
        agg = results["aggregate_oos"]
        print(f"  OOS: Sharpe={agg.get('mean_sharpe', 0):+.3f} PnL={agg.get('mean_pnl_pct', 0):+.1f}%")
    if results.get("holdout") and results["holdout"].get("mean"):
        ho = results["holdout"]["mean"]
        print(f"  Holdout: Sharpe={ho.get('sharpe', 0):+.3f} PnL={ho.get('pnl_pct', 0):+.1f}%")

    out_path = RESULTS_DIR / f"rl_btc_{algo_name.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RL V6 — The MACD Filter")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0, help="Override training steps for non-quick runs")
    args = parser.parse_args()
    run_training(
        seeds=SEEDS[:args.seeds],
        quick=args.quick,
        train_steps_override=(args.steps or None),
    )


if __name__ == "__main__":
    main()
