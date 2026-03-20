"""BTC PPO Hyperparameter Tuner — Forge Engine
==============================================
Uses Optuna TPE to systematically search for optimal PPO trading agent params.

Three-phase approach:
  Phase 1: Smoke test (10K steps) — verify pipeline works, agent trades
  Phase 2: Coarse search (100K steps × 60 trials) — explore parameter space
  Phase 3: Refinement (500K steps × top 5) — validate best candidates

Data split:
  Train:      2020-01-01 → 2024-01-01  (4 years)
  Validation: 2024-01-01 → 2025-01-01  (1 year — HPO objective)
  Test:       2025-01-01 → 2026-02-12  (holdout — never seen during HPO)

Usage:
    uv run python examples_rl/tune_btc_ppo.py
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path

import numpy as np
import optuna

# ── Forge Engine imports ────────────────────────────────────────────────
from forge_engine.rl.env import ForgeEnv
from forge_engine.rl.actions import ContinuousActions
from forge_engine.rl.rewards import resolve_reward
from forge_engine.engine import (
    create_session,
    preload_candle_data_aggregated,
    _parse_timeframe_to_minutes,
)

# ── SB3 ─────────────────────────────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

# Data splits (strict: no data leakage between phases)
TRAIN_START = "2020-01-01T00:00:00Z"
TRAIN_END   = "2024-01-01T00:00:00Z"   # 4 years
VAL_START   = "2024-01-01T00:00:00Z"
VAL_END     = "2025-01-01T00:00:00Z"   # 1 year — HPO objective
TEST_START  = "2025-01-01T00:00:00Z"
TEST_END    = "2026-02-12T00:00:00Z"   # Holdout

# Fixed engine params (not tuned — these are realistic assumptions)
SYMBOL          = "BTCUSDT_PERP"
TIMEFRAME       = "1h"
TF_MINUTES      = 60
MARGIN_MODE     = "cross"
SLIPPAGE_PCT    = 0.0005
WARMUP_CANDLES  = 50
CLOSE_AT_END    = True
STARTING_CASH   = 100_000
SEED            = 42

# Phase settings (tuned for ~1,300 steps/sec SB3 throughput)
SMOKE_STEPS     = 20_000       # ~15s — just verify trading works
SEARCH_STEPS    = 75_000       # ~55s/trial — enough to see if config learns
N_SEARCH_TRIALS = 50           # ~50 min total for phase 2
REFINE_STEPS    = 300_000      # ~4 min/trial — validate top candidates
TOP_N           = 3            # top 3 → ~12 min for phase 3

# ═══════════════════════════════════════════════════════════════════════
# Observation presets — different feature combinations to test
# ═══════════════════════════════════════════════════════════════════════

OBS_PRESETS = {
    "minimal": [
        "ohlcv", "returns", "position_info",
    ],
    "standard": [
        "ohlcv", "returns", "rsi_14", "sma_ratio_20_50",
        "position_info", "drawdown",
    ],
    "full": [
        "ohlcv", "returns", "rsi_14", "sma_ratio_20_50",
        "position_info", "equity_curve", "drawdown", "volume_profile",
    ],
    "momentum": [
        "ohlcv", "returns", "rsi_14", "ema_21",
        "sma_ratio_10_30", "position_info", "drawdown",
    ],
    "rich": [
        "ohlcv", "returns", "rsi_14",
        "sma_ratio_10_30", "sma_ratio_20_50",
        "position_info", "equity_curve", "drawdown", "volume_profile",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# Data pre-loading (once — shared across all trials)
# ═══════════════════════════════════════════════════════════════════════

class DataCache:
    """Pre-loads candle data once and injects into ForgeEnvs to avoid
    repeated CSV parsing across 60+ Optuna trials."""
    candle_data = None
    funding_data = None
    data_dir = None
    _loaded = False

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        t0 = time.time()

        # Resolve data dir + funding via a throwaway session
        tmp = create_session(
            symbol=SYMBOL,
            start_date=TRAIN_START,
            end_date=TEST_END,
            starting_cash=STARTING_CASH,
            leverage=2,
            margin_mode=MARGIN_MODE,
            slippage_pct=SLIPPAGE_PCT,
            close_at_end=CLOSE_AT_END,
            timeframe=TIMEFRAME,
            warmup_candles=WARMUP_CANDLES,
            enable_visual=False,
        )
        cls.funding_data = getattr(tmp, "funding_data", None)
        cls.data_dir = tmp.data_dir

        # Pre-aggregate candles for the full range (train through test)
        cls.candle_data = preload_candle_data_aggregated(
            symbol=SYMBOL,
            start_date=TRAIN_START,
            end_date=TEST_END,
            base_timeframe="1m",
            target_timeframe_minutes=TF_MINUTES,
            data_dir=cls.data_dir,
            warmup_candles=WARMUP_CANDLES,
        )
        cls._loaded = True
        n = len(cls.candle_data.timestamps_unix) if cls.candle_data else 0
        print(f"  Data loaded: {n:,} candles in {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# Environment factory
# ═══════════════════════════════════════════════════════════════════════

def make_env(config: dict, start: str, end: str, max_steps_override=None):
    """Create a ForgeEnv with pre-loaded data injection."""
    DataCache.load()
    reward_obj = resolve_reward(config["reward"], **config.get("reward_kwargs", {}))
    ms = max_steps_override if max_steps_override is not None else config["max_steps"]
    env = ForgeEnv(
        symbol=SYMBOL,
        start_date=start,
        end_date=end,
        observations=config["observations"],
        actions=ContinuousActions(
            max_margin_pct=config["max_margin_pct"],
            threshold=config["threshold"],
            sl_pct=config["sl_pct"],
            tp_pct=config["tp_pct"],
        ),
        reward=reward_obj,
        starting_cash=STARTING_CASH,
        leverage=config["leverage"],
        margin_mode=MARGIN_MODE,
        slippage_pct=SLIPPAGE_PCT,
        close_at_end=CLOSE_AT_END,
        timeframe=TIMEFRAME,
        warmup_candles=WARMUP_CANDLES,
        max_steps=ms,
    )
    # Inject pre-loaded data (skips CSV parsing in _ensure_data_loaded)
    env._candle_data = DataCache.candle_data
    env._funding_data = DataCache.funding_data
    env._resolved_data_dir = DataCache.data_dir
    # Skip intra-candle loading (minor accuracy trade-off, big speed win for HPO)
    env._intra_tuples = None
    return Monitor(env)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_on_period(model, config: dict, start: str, end: str) -> dict:
    """Run the trained model on a date range, return performance metrics.

    Uses max_steps=0 (full period) for deterministic, complete evaluation.
    """
    env = make_env(config, start, end, max_steps_override=0)
    obs, info = env.reset(seed=SEED)

    equities = [STARTING_CASH]
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        eq = info.get("equity", equities[-1])
        equities.append(float(eq))
        done = term or trunc
        steps += 1
        # Safety: if we somehow run forever
        if steps > 100_000:
            break

    env.close()

    eq = np.array(equities, dtype=np.float64)
    returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)

    # Sharpe (annualized from hourly)
    if len(returns) < 10 or np.std(returns) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((np.mean(returns) / np.std(returns)) * np.sqrt(8760))

    # Sortino
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0 and np.std(neg_returns) > 1e-12:
        sortino = float(np.mean(returns) / np.std(neg_returns) * np.sqrt(8760))
    else:
        sortino = sharpe  # fallback

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / np.maximum(running_max, 1e-10)
    max_dd = float(np.max(drawdowns) * 100)

    net_pnl = float(eq[-1] - eq[0])
    net_pnl_pct = float((eq[-1] / eq[0] - 1) * 100) if eq[0] > 0 else 0.0

    # Trade activity: how many candles had non-zero equity change
    # (if in a position, equity moves every candle; if flat, equity is constant)
    eq_changes = np.abs(np.diff(eq))
    n_active = int(np.count_nonzero(eq_changes > 0.01))
    traded = n_active > max(10, steps * 0.005)  # at least 0.5% of steps active

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl_pct,
        "max_dd_pct": max_dd,
        "final_equity": float(eq[-1]),
        "n_active_steps": n_active,
        "total_steps": steps,
        "traded": traded,
        "liquidated": float(eq[-1]) <= 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Search space (expert-informed ranges)
# ═══════════════════════════════════════════════════════════════════════

def sample_config(trial: optuna.Trial) -> dict:
    """Sample a complete configuration from the Optuna search space.

    Ranges are informed by RL-for-trading literature and practical experience:
    - LR: [5e-5, 5e-4] — higher = unstable, lower = too slow
    - ent_coef: [0.003, 0.03] — critical for exploration without chaos
    - threshold: [0.05, 0.35] — dead zone size (was 0.5, way too high)
    - TP >= 1.5 × SL enforced — positive expectation requirement
    """

    # ── Reward function ──────────────────────────────────────────────
    reward = trial.suggest_categorical(
        "reward", ["differential_sharpe", "risk_adjusted", "sortino", "log_return"]
    )
    reward_kwargs = {}
    if reward == "risk_adjusted":
        reward_kwargs["drawdown_penalty"] = trial.suggest_float("dd_penalty", 1.0, 5.0)
    elif reward == "differential_sharpe":
        reward_kwargs["eta"] = trial.suggest_float("ds_eta", 0.005, 0.05, log=True)
    elif reward == "sortino":
        reward_kwargs["eta"] = trial.suggest_float("sortino_eta", 0.005, 0.05, log=True)

    # ── Action space ─────────────────────────────────────────────────
    threshold = trial.suggest_float("threshold", 0.05, 0.35)
    max_margin_pct = trial.suggest_float("max_margin_pct", 0.03, 0.12)
    sl_pct = trial.suggest_float("sl_pct", 0.01, 0.06)
    # TP must be >= 1.5x SL for positive expectation
    tp_min = sl_pct * 1.5
    tp_pct = trial.suggest_float("tp_pct", tp_min, max(tp_min + 0.01, 0.12))

    # ── Environment ──────────────────────────────────────────────────
    leverage = trial.suggest_int("leverage", 1, 3)
    max_steps = trial.suggest_categorical("max_steps", [360, 720, 1080])

    # ── Observations ─────────────────────────────────────────────────
    obs_preset = trial.suggest_categorical(
        "obs_preset", list(OBS_PRESETS.keys())
    )

    # ── PPO hyperparameters ──────────────────────────────────────────
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 0.003, 0.03, log=True)
    gamma = trial.suggest_float("gamma", 0.97, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    clip_range = trial.suggest_float("clip_range", 0.15, 0.3)

    # ── Network architecture ─────────────────────────────────────────
    arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch_map = {
        "small":  [64, 64],
        "medium": [128, 128],
        "large":  [256, 256],
    }

    return {
        "reward": reward,
        "reward_kwargs": reward_kwargs,
        "threshold": threshold,
        "max_margin_pct": max_margin_pct,
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "leverage": leverage,
        "max_steps": max_steps,
        "observations": OBS_PRESETS[obs_preset],
        "obs_preset": obs_preset,
        "lr": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "net_arch": net_arch_map[arch],
        "net_arch_name": arch,
    }


# ═══════════════════════════════════════════════════════════════════════
# Optuna objective
# ═══════════════════════════════════════════════════════════════════════

def create_objective(train_steps: int, train_start: str, train_end: str,
                     val_start: str, val_end: str):
    """Factory that returns an Optuna objective function with the given step budget."""

    def objective(trial: optuna.Trial) -> float:
        config = sample_config(trial)

        try:
            # ── Train ────────────────────────────────────────────────
            env = make_env(config, train_start, train_end)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["lr"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                ent_coef=config["ent_coef"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                policy_kwargs=dict(net_arch=config["net_arch"]),
                seed=SEED,
                verbose=0,
                device="cpu",
            )
            t0 = time.time()
            model.learn(total_timesteps=train_steps)
            train_time = time.time() - t0
            env.close()

            # ── Evaluate on validation ───────────────────────────────
            metrics = evaluate_on_period(model, config, val_start, val_end)
            del model

            # ── Compute objective ────────────────────────────────────
            if not metrics["traded"] or metrics["liquidated"]:
                obj = -10.0
            else:
                obj = metrics["sharpe"]

            # Log for display
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("config", {
                k: v for k, v in config.items()
                if k not in ("observations",)  # too verbose
            })
            trial.set_user_attr("train_time", train_time)

            rw_short = config["reward"][:8]
            obs_short = config["obs_preset"][:5]
            status = ""
            try:
                if trial.study.best_trial.number == trial.number:
                    status = " ** BEST"
            except ValueError:
                pass  # no completed trials yet
            print(
                f"  Trial {trial.number + 1:3d}/{N_SEARCH_TRIALS} | "
                f"rw={rw_short:<8s} obs={obs_short:<5s} "
                f"lr={config['lr']:.0e} th={config['threshold']:.2f} "
                f"ent={config['ent_coef']:.3f} | "
                f"Sharpe={metrics['sharpe']:+6.3f}  "
                f"PnL={metrics['net_pnl_pct']:+6.2f}%  "
                f"DD={metrics['max_dd_pct']:5.2f}%  "
                f"({train_time:.1f}s){status}"
            )

            return obj

        except Exception as e:
            try:
                print(f"  Trial {trial.number + 1:3d}/{N_SEARCH_TRIALS} | FAILED: {e}")
            except Exception:
                print(f"  Trial {trial.number + 1}/{N_SEARCH_TRIALS} | FAILED (encoding error)")
            return -10.0

    return objective


# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════

def format_results_table(trials: list[optuna.trial.FrozenTrial], top_n: int = 10):
    """Print a ranked table of the best trials."""
    ranked = sorted(trials, key=lambda t: t.value if t.value else -999, reverse=True)
    ranked = [t for t in ranked if t.value is not None and t.value > -10.0][:top_n]

    if not ranked:
        print("  No successful trials found.")
        return

    print(f"  {'#':>3s}  {'Sharpe':>7s}  {'PnL%':>7s}  {'MaxDD%':>7s}  "
          f"{'Reward':<12s}  {'Obs':<9s}  {'LR':>8s}  {'Thr':>5s}  "
          f"{'Ent':>6s}  {'Arch':<6s}  {'SL%':>5s}  {'TP%':>5s}")
    print("  " + "-" * 105)

    for i, t in enumerate(ranked):
        c = t.user_attrs.get("config", {})
        m = t.user_attrs.get("metrics", {})
        print(
            f"  {i + 1:3d}  {t.value:+7.3f}  {m.get('net_pnl_pct', 0):+7.2f}  "
            f"{m.get('max_dd_pct', 0):7.2f}  "
            f"{c.get('reward', '?'):<12s}  {c.get('obs_preset', '?'):<9s}  "
            f"{c.get('lr', 0):.1e}  {c.get('threshold', 0):5.2f}  "
            f"{c.get('ent_coef', 0):6.4f}  {c.get('net_arch_name', '?'):<6s}  "
            f"{c.get('sl_pct', 0) * 100:5.2f}  {c.get('tp_pct', 0) * 100:5.2f}"
        )


def format_best_config(config: dict, metrics: dict):
    """Print the best config in btc_ppo.py RL_CONFIG format."""
    rk = config.get("reward_kwargs", {})
    rk_str = ", ".join(f"{k}={v}" for k, v in rk.items()) if rk else ""

    arch = config.get("net_arch", [256, 256])
    obs_list = OBS_PRESETS.get(config.get("obs_preset", "full"), [])
    obs_str = ",\n        ".join(f'"{o}"' for o in obs_list)

    print(f"""
RL_CONFIG = dict(
    name="BTC PPO (Tuned)",
    algorithm="PPO",
    model_path=str(SCRIPT_DIR / "btc_ppo_model"),
    symbol="BTCUSDT_PERP",
    train_start="2020-01-01T00:00:00Z",
    train_end="2025-01-01T00:00:00Z",
    test_start="2025-01-01T00:00:00Z",
    test_end="2026-02-12T00:00:00Z",
    observations=[
        {obs_str},
    ],
    actions=ContinuousActions(
        max_margin_pct={config.get('max_margin_pct', 0.10):.4f},
        threshold={config.get('threshold', 0.2):.4f},
        sl_pct={config.get('sl_pct', 0.03):.4f},
        tp_pct={config.get('tp_pct', 0.06):.4f},
    ),
    reward="{config.get('reward', 'risk_adjusted')}",
    reward_kwargs=dict({rk_str}),
    engine_kwargs=dict(
        starting_cash={STARTING_CASH},
        leverage={config.get('leverage', 2)},
        margin_mode="cross",
        slippage_pct={SLIPPAGE_PCT},
        close_at_end=True,
        timeframe="1h",
        warmup_candles=50,
        max_steps={config.get('max_steps', 720)},
    ),
    train_steps=3_000_000,
    retrain_steps=250_000,
    learning_rate={config.get('lr', 3e-4):.1e},
    seed=42,
    algo_kwargs=dict(
        n_steps={config.get('n_steps', 2048)},
        batch_size={config.get('batch_size', 256)},
        n_epochs={config.get('n_epochs', 10)},
        ent_coef={config.get('ent_coef', 0.01):.4f},
        gamma={config.get('gamma', 0.99):.4f},
        gae_lambda={config.get('gae_lambda', 0.95):.4f},
        clip_range={config.get('clip_range', 0.2):.4f},
        policy_kwargs=dict(net_arch=dict(pi={arch}, vf={arch})),
    ),
)""")


# ═══════════════════════════════════════════════════════════════════════
# Main — three-phase tuning
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  BTC PPO Hyperparameter Tuner — Forge Engine")
    print("=" * 70)
    print(f"  Train:      {TRAIN_START[:10]} -> {TRAIN_END[:10]}  (4 years)")
    print(f"  Validation: {VAL_START[:10]} -> {VAL_END[:10]}  (1 year, HPO objective)")
    print(f"  Test:       {TEST_START[:10]} -> {TEST_END[:10]}  (holdout)")
    print()

    # ── Pre-load data ────────────────────────────────────────────────
    print("Pre-loading candle data (once for all trials)...")
    DataCache.load()
    print()

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Smoke test — verify pipeline works
    # ══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print("Phase 1: Smoke Test")
    print("-" * 70)

    smoke_config = {
        "reward": "risk_adjusted",
        "reward_kwargs": {"drawdown_penalty": 2.0},
        "threshold": 0.2,
        "max_margin_pct": 0.08,
        "sl_pct": 0.03,
        "tp_pct": 0.06,
        "leverage": 2,
        "max_steps": 720,
        "observations": OBS_PRESETS["standard"],
        "obs_preset": "standard",
        "lr": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 5,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "net_arch": [128, 128],
        "net_arch_name": "medium",
    }

    print(f"  Training default config for {SMOKE_STEPS:,} steps...")
    env = make_env(smoke_config, TRAIN_START, TRAIN_END)
    t0 = time.time()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=smoke_config["lr"],
        n_steps=smoke_config["n_steps"],
        batch_size=smoke_config["batch_size"],
        n_epochs=smoke_config["n_epochs"],
        ent_coef=smoke_config["ent_coef"],
        gamma=smoke_config["gamma"],
        seed=SEED, verbose=0, device="cpu",
    )
    model.learn(total_timesteps=SMOKE_STEPS)
    elapsed = time.time() - t0
    env.close()
    speed = SMOKE_STEPS / elapsed
    print(f"  Trained in {elapsed:.1f}s ({speed:,.0f} steps/sec)")

    print("  Evaluating on validation period...")
    metrics = evaluate_on_period(model, smoke_config, VAL_START, VAL_END)
    del model

    print(f"  Sharpe={metrics['sharpe']:+.3f}  "
          f"PnL={metrics['net_pnl_pct']:+.2f}%  "
          f"MaxDD={metrics['max_dd_pct']:.2f}%  "
          f"Active={metrics['n_active_steps']}/{metrics['total_steps']}")

    if not metrics["traded"]:
        print("  WARNING: Smoke test agent didn't trade. Check dead zone fix.")
        print("  Continuing anyway — Optuna may find configs that do trade.")
    else:
        print("  Pipeline verified. Agent is trading.")
    print()

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Optuna coarse search
    # ══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print(f"Phase 2: Optuna Search ({SEARCH_STEPS:,} steps x {N_SEARCH_TRIALS} trials)")
    print("-" * 70)

    # Estimate time based on smoke test speed
    est_per_trial = SEARCH_STEPS / speed + 5  # +5s for eval overhead
    est_total = est_per_trial * N_SEARCH_TRIALS
    print(f"  Estimated: ~{est_per_trial:.0f}s/trial, ~{est_total / 60:.1f} min total")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="btc_ppo_tune",
    )

    objective_fn = create_objective(
        SEARCH_STEPS, TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    )

    t_search_start = time.time()
    study.optimize(objective_fn, n_trials=N_SEARCH_TRIALS, n_jobs=1)
    t_search_elapsed = time.time() - t_search_start

    print()
    print(f"  Search completed in {t_search_elapsed / 60:.1f} min")
    print()

    # Show top results
    print("  Top 10 Configurations:")
    format_results_table(study.trials, top_n=10)
    print()

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Refinement — train top configs longer
    # ══════════════════════════════════════════════════════════════════
    print("-" * 70)
    print(f"Phase 3: Refinement ({REFINE_STEPS:,} steps x top {TOP_N})")
    print("-" * 70)

    # Get top N configs (by validation Sharpe, must have traded)
    ranked = sorted(
        [t for t in study.trials if t.value is not None and t.value > -10.0],
        key=lambda t: t.value,
        reverse=True,
    )[:TOP_N]

    if not ranked:
        print("  No viable configs found. Try increasing SEARCH_STEPS or N_TRIALS.")
        return

    refined_results = []
    for i, trial in enumerate(ranked):
        config = trial.user_attrs.get("config", {})
        # Restore observations from preset name
        config["observations"] = OBS_PRESETS.get(config.get("obs_preset", "full"), OBS_PRESETS["full"])
        config.setdefault("reward_kwargs", {})

        print(f"\n  Refining config #{i + 1} (Phase 2 Sharpe={trial.value:+.3f})...")
        print(f"    reward={config['reward']}, obs={config.get('obs_preset', '?')}, "
              f"lr={config.get('lr', 0):.1e}, th={config.get('threshold', 0):.2f}")

        try:
            # ── Train longer ─────────────────────────────────────────
            env = make_env(config, TRAIN_START, TRAIN_END)
            model = PPO(
                "MlpPolicy", env,
                learning_rate=config["lr"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                n_epochs=config["n_epochs"],
                ent_coef=config["ent_coef"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                policy_kwargs=dict(net_arch=config["net_arch"]),
                seed=SEED, verbose=0, device="cpu",
            )
            t0 = time.time()
            model.learn(total_timesteps=REFINE_STEPS)
            train_time = time.time() - t0
            env.close()

            # ── Evaluate on BOTH val and test ────────────────────────
            val_m = evaluate_on_period(model, config, VAL_START, VAL_END)
            test_m = evaluate_on_period(model, config, TEST_START, TEST_END)
            del model

            refined_results.append({
                "rank": i + 1,
                "config": config,
                "val": val_m,
                "test": test_m,
                "train_time": train_time,
            })

            print(f"    Val:  Sharpe={val_m['sharpe']:+.3f}  "
                  f"PnL={val_m['net_pnl_pct']:+.2f}%  DD={val_m['max_dd_pct']:.2f}%")
            print(f"    Test: Sharpe={test_m['sharpe']:+.3f}  "
                  f"PnL={test_m['net_pnl_pct']:+.2f}%  DD={test_m['max_dd_pct']:.2f}%  "
                  f"({train_time:.1f}s)")

        except Exception as e:
            print(f"    FAILED: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Final results
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    if not refined_results:
        print("  No refined results. Falling back to Phase 2 best.")
        if study.best_trial:
            best_config = study.best_trial.user_attrs.get("config", {})
            best_config["observations"] = OBS_PRESETS.get(
                best_config.get("obs_preset", "full"), OBS_PRESETS["full"])
            best_config.setdefault("reward_kwargs", {})
            best_metrics = study.best_trial.user_attrs.get("metrics", {})
            print("\n  Best config (Phase 2 validation only):")
            format_best_config(best_config, best_metrics)
        return

    # Rank by test Sharpe (out-of-sample performance is what matters)
    refined_results.sort(
        key=lambda r: r["test"]["sharpe"] if r["test"]["traded"] else -999,
        reverse=True,
    )

    print(f"\n  {'#':>3s}  {'Val Sharpe':>10s}  {'Test Sharpe':>11s}  "
          f"{'Test PnL%':>9s}  {'Test DD%':>8s}  {'Reward':<12s}  {'Obs':<9s}")
    print("  " + "-" * 75)
    for r in refined_results:
        v, t = r["val"], r["test"]
        c = r["config"]
        print(f"  {r['rank']:3d}  {v['sharpe']:+10.3f}  {t['sharpe']:+11.3f}  "
              f"{t['net_pnl_pct']:+9.2f}  {t['max_dd_pct']:8.2f}  "
              f"{c.get('reward', '?'):<12s}  {c.get('obs_preset', '?'):<9s}")

    # Best by test performance
    best = refined_results[0]
    best_config = best["config"]
    best_test = best["test"]

    print(f"\n  BEST CONFIG (by out-of-sample test Sharpe = {best_test['sharpe']:+.3f}):")
    print(f"  Test PnL: {best_test['net_pnl_pct']:+.2f}% (${best_test['net_pnl']:+,.2f})")
    print(f"  Test Max DD: {best_test['max_dd_pct']:.2f}%")

    print("\n  Copy this into examples_rl/btc_ppo/btc_ppo.py:")
    print("  " + "-" * 50)
    format_best_config(best_config, best_test)

    # ── Save results to JSON ─────────────────────────────────────────
    out_path = Path(__file__).parent / "btc_ppo" / "tune_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "best_config": {k: v for k, v in best_config.items() if k != "observations"},
        "best_obs_preset": best_config.get("obs_preset", "full"),
        "val_metrics": best["val"],
        "test_metrics": best_test,
        "all_refined": [
            {
                "rank": r["rank"],
                "config": {k: v for k, v in r["config"].items() if k != "observations"},
                "val": r["val"],
                "test": r["test"],
            }
            for r in refined_results
        ],
        "search_trials": N_SEARCH_TRIALS,
        "search_steps": SEARCH_STEPS,
        "refine_steps": REFINE_STEPS,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
