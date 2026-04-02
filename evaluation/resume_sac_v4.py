"""Resume SAC V4 training from fold 4, seed 123.
Loads partial results, completes remaining folds + holdout, saves final file.

Usage:
    uv run python evaluation/resume_sac_v4.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge_engine.rl.env import ForgeEnv
from forge_engine.rl.actions import ContinuousActions
from forge_engine.rl.rewards import resolve_reward
from forge_engine.engine import (
    create_session, preload_candle_data_aggregated, _isoformat_z,
)
from forge_engine.optuna_optimizer import WalkForwardSplitter, WalkForwardConfig
from evaluation.artifacts import build_model_eval_env, evaluate_rl_model_on_period

from stable_baselines3 import SAC
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
PARTIAL_FILE = RESULTS_DIR / "rl_btc_sac_v4_partial.json"
FINAL_FILE = RESULTS_DIR / "rl_btc_sac_v4.json"

OBS = ["ohlcv", "returns", "rsi_14", "atr_14", "sma_ratio_20_50", "position_info", "drawdown"]
REWARD = "differential_sharpe"
REWARD_KWARGS = dict(eta=0.01, scale=1.0)
ENGINE_KWARGS = dict(
    starting_cash=STARTING_CASH, leverage=3, margin_mode=MARGIN_MODE,
    slippage_pct=SLIPPAGE_PCT, close_at_end=CLOSE_AT_END,
    timeframe=TIMEFRAME, warmup_candles=WARMUP_CANDLES, max_steps=720,
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


def make_env(start_iso, end_iso, max_steps_override=None):
    DataCache.load()
    reward_obj = resolve_reward(REWARD, **REWARD_KWARGS)
    ek = ENGINE_KWARGS.copy()
    if max_steps_override is not None:
        ek["max_steps"] = max_steps_override
    env = ForgeEnv(
        symbol=SYMBOL, start_date=start_iso, end_date=end_iso,
        observations=OBS,
        actions=ContinuousActions(max_margin_pct=0.15, threshold=0.20, sl_pct=0.03, tp_pct=0.09),
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
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)


def make_model(env, seed):
    return SAC(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=300_000,
        learning_starts=10_000, batch_size=256, tau=0.005, gamma=0.99,
        ent_coef="auto", seed=seed, verbose=0, device=SB3_DEVICE,
        policy_kwargs=dict(net_arch=dict(pi=[64, 32], qf=[64, 32])),
    )


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


def main():
    print("=" * 70)
    print("  SAC V4 RESUME — picking up from fold 4, seed 123")
    print("=" * 70)

    # Load partial results
    with open(PARTIAL_FILE) as f:
        results = json.load(f)

    # Remove status fields
    results.pop("status", None)
    results.pop("resume_from", None)
    results["algorithm"] = "SAC_v4"

    # Generate splits (same as original)
    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=N_SPLITS, test_ratio=TEST_RATIO, mode=WFA_MODE,
    ))
    splits, holdout = splitter.generate_splits(
        full_start=FULL_START, full_end=FULL_END,
        holdout_ratio=HOLDOUT_RATIO, timeframe_minutes=TF_MINUTES,
    )

    DataCache.load()
    t_total_start = time.time()

    # ── Complete fold 4 (seeds 123, 456) ──────────────────────────────
    fold4 = results["folds"][3]  # index 3 = fold 4
    fold4.pop("status", None)
    tr_s, tr_e, te_s, te_e = splits[3]

    print(f"\n  Resuming Fold 4: Test {dt_to_datestr(te_s)} -> {dt_to_datestr(te_e)}")
    for seed in [123, 456]:
        t0 = time.time()
        print(f"    Seed (seed={seed})... ", end="", flush=True)
        try:
            env = make_train_env(dt_to_iso(tr_s), dt_to_iso(tr_e))
            model = make_model(env, seed)
            model.learn(total_timesteps=TRAIN_STEPS)
            env.close()
            metrics = evaluate_on_period(model, dt_to_iso(te_s), dt_to_iso(te_e), seed=seed)
            fold4["seeds"][str(seed)] = metrics
            print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  DD={metrics['max_dd_pct']:.1f}%  ({time.time()-t0:.0f}s)")
            del model
        except Exception as e:
            print(f"FAILED: {e}")
            fold4["seeds"][str(seed)] = {"error": str(e)}

    # Compute fold 4 mean/std
    seed_metrics = [v for v in fold4["seeds"].values() if "error" not in v]
    if seed_metrics:
        for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
            vals = [m[key] for m in seed_metrics if key in m]
            if vals:
                fold4["mean"][key] = round(float(np.mean(vals)), 4)
                fold4["std"][key] = round(float(np.std(vals)), 4)
        print(f"    Fold 4 mean: Sharpe={fold4['mean'].get('sharpe', 0):+.3f}")

    # Save checkpoint
    results["folds"][3] = fold4
    with open(PARTIAL_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  [checkpoint saved]")

    # ── Fold 5 ────────────────────────────────────────────────────────
    tr_s, tr_e, te_s, te_e = splits[4]
    print(f"\n  Fold 5: Train {dt_to_datestr(tr_s)} -> {dt_to_datestr(tr_e)} | Test {dt_to_datestr(te_s)} -> {dt_to_datestr(te_e)}")

    fold5 = {
        "fold": 5,
        "train_period": [dt_to_datestr(tr_s), dt_to_datestr(tr_e)],
        "test_period": [dt_to_datestr(te_s), dt_to_datestr(te_e)],
        "seeds": {}, "mean": {}, "std": {},
    }
    seed_metrics_list = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"    Seed (seed={seed})... ", end="", flush=True)
        try:
            env = make_train_env(dt_to_iso(tr_s), dt_to_iso(tr_e))
            model = make_model(env, seed)
            model.learn(total_timesteps=TRAIN_STEPS)
            env.close()
            metrics = evaluate_on_period(model, dt_to_iso(te_s), dt_to_iso(te_e), seed=seed)
            fold5["seeds"][str(seed)] = metrics
            seed_metrics_list.append(metrics)
            print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  DD={metrics['max_dd_pct']:.1f}%  ({time.time()-t0:.0f}s)")
            del model
        except Exception as e:
            print(f"FAILED: {e}")
            fold5["seeds"][str(seed)] = {"error": str(e)}

    if seed_metrics_list:
        for key in ["sharpe", "sortino", "max_dd_pct", "pnl_pct"]:
            vals = [m[key] for m in seed_metrics_list if key in m]
            if vals:
                fold5["mean"][key] = round(float(np.mean(vals)), 4)
                fold5["std"][key] = round(float(np.std(vals)), 4)
        print(f"    Fold 5 mean: Sharpe={fold5['mean'].get('sharpe', 0):+.3f}")

    if len(results["folds"]) < 5:
        results["folds"].append(fold5)
    else:
        results["folds"][4] = fold5

    # Save checkpoint
    with open(PARTIAL_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  [checkpoint saved]")

    # ── Holdout ───────────────────────────────────────────────────────
    if holdout:
        ho_s_iso, ho_e_iso = dt_to_iso(holdout[0]), dt_to_iso(holdout[1])
        print(f"\n  Holdout: {dt_to_datestr(holdout[0])} -> {dt_to_datestr(holdout[1])}")

        holdout_result = {
            "period": [dt_to_datestr(holdout[0]), dt_to_datestr(holdout[1])],
            "seeds": {}, "mean": {}, "std": {},
        }
        ho_metrics_list = []
        for seed in SEEDS:
            t0 = time.time()
            print(f"    Seed (seed={seed})... ", end="", flush=True)
            try:
                env = make_train_env(dt_to_iso(FULL_START), ho_s_iso)
                model = make_model(env, seed)
                model.learn(total_timesteps=TRAIN_STEPS)
                env.close()
                metrics = evaluate_on_period(model, ho_s_iso, ho_e_iso, seed=seed)
                holdout_result["seeds"][str(seed)] = metrics
                ho_metrics_list.append(metrics)
                print(f"Sharpe={metrics['sharpe']:+.3f}  PnL={metrics['pnl_pct']:+.1f}%  DD={metrics['max_dd_pct']:.1f}%  ({time.time()-t0:.0f}s)")
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

    # ── Aggregate OOS ─────────────────────────────────────────────────
    all_test_metrics = []
    for fold in results["folds"]:
        for seed_data in fold.get("seeds", {}).values():
            if isinstance(seed_data, dict) and "error" not in seed_data and "sharpe" in seed_data:
                all_test_metrics.append(seed_data)

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
    print(f"  SAC V4 COMPLETE -- {total_elapsed/60:.1f} min (resume portion)")
    print("=" * 70)
    if results["aggregate_oos"]:
        agg = results["aggregate_oos"]
        print(f"  Aggregate OOS: Sharpe={agg.get('mean_sharpe', 0):+.3f} "
              f"(+/-{agg.get('std_sharpe', 0):.3f})")
    if results.get("holdout") and results["holdout"].get("mean"):
        ho = results["holdout"]["mean"]
        print(f"  Holdout: Sharpe={ho.get('sharpe', 0):+.3f}  PnL={ho.get('pnl_pct', 0):+.1f}%")

    # Save final results
    with open(FINAL_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Final results saved to {FINAL_FILE}")


if __name__ == "__main__":
    main()
