"""Training pipeline for RL agents using stable-baselines3.

Provides a simple interface to train PPO, DQN, A2C, and SAC agents
on a ForgeEnv environment with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np


# Supported algorithms and their SB3 classes
_ALGO_MAP = {
    "PPO": "stable_baselines3.PPO",
    "DQN": "stable_baselines3.DQN",
    "A2C": "stable_baselines3.A2C",
}


def _get_algo_class(algorithm: str):
    """Lazy-import the SB3 algorithm class."""
    name = algorithm.upper()
    if name not in _ALGO_MAP:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. Supported: {list(_ALGO_MAP.keys())}"
        )

    if name == "PPO":
        from stable_baselines3 import PPO

        return PPO
    elif name == "DQN":
        from stable_baselines3 import DQN

        return DQN
    elif name == "A2C":
        from stable_baselines3 import A2C

        return A2C
    else:
        raise ValueError(f"Unknown algorithm: '{algorithm}'")


def train_agent(
    env,
    algorithm: str = "DQN",
    total_timesteps: int = 100_000,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    policy: str = "MlpPolicy",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    learning_rate: float = 3e-4,
    callback: Optional[Any] = None,
    **algo_kwargs,
):
    """Train an RL agent on a ForgeEnv environment.

    Parameters:
        env: ForgeEnv instance (or any Gymnasium env)
        algorithm: "DQN", "PPO", or "A2C"
        total_timesteps: Total training steps
        seed: Random seed for reproducibility
        save_path: Path to save the trained model (e.g., "models/btc_dqn")
        tensorboard_log: Directory for TensorBoard logs
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        policy: Policy type (default "MlpPolicy")
        policy_kwargs: Extra policy arguments (e.g., net_arch)
        learning_rate: Learning rate
        callback: Optional SB3 callback (or list of callbacks) for training
        **algo_kwargs: Extra arguments passed to the SB3 algorithm constructor

    Returns:
        The trained SB3 model instance
    """
    AlgoClass = _get_algo_class(algorithm)

    # Build constructor args
    kwargs: Dict[str, Any] = {
        "policy": policy,
        "env": env,
        "learning_rate": learning_rate,
        "verbose": verbose,
        "seed": seed,
    }

    if tensorboard_log:
        kwargs["tensorboard_log"] = tensorboard_log

    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs

    # Algorithm-specific defaults
    algo_name = algorithm.upper()
    if algo_name == "DQN":
        kwargs.setdefault("buffer_size", algo_kwargs.pop("buffer_size", 50_000))
        kwargs.setdefault("learning_starts", algo_kwargs.pop("learning_starts", 1000))
        kwargs.setdefault("batch_size", algo_kwargs.pop("batch_size", 64))
        kwargs.setdefault(
            "exploration_fraction", algo_kwargs.pop("exploration_fraction", 0.2)
        )
        kwargs.setdefault(
            "exploration_final_eps", algo_kwargs.pop("exploration_final_eps", 0.05)
        )
    elif algo_name == "PPO":
        kwargs.setdefault("n_steps", algo_kwargs.pop("n_steps", 2048))
        kwargs.setdefault("batch_size", algo_kwargs.pop("batch_size", 64))
        kwargs.setdefault("n_epochs", algo_kwargs.pop("n_epochs", 10))
        kwargs.setdefault("gamma", algo_kwargs.pop("gamma", 0.99))
        kwargs.setdefault("gae_lambda", algo_kwargs.pop("gae_lambda", 0.95))
        kwargs.setdefault("clip_range", algo_kwargs.pop("clip_range", 0.2))
        kwargs.setdefault("ent_coef", algo_kwargs.pop("ent_coef", 0.01))

    # Pass through remaining kwargs
    kwargs.update(algo_kwargs)

    # Create and train
    model = AlgoClass(**kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save if requested
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)
        if verbose >= 1:
            print(f"Model saved to: {save_path}")

    return model


def load_agent(path: str, algorithm: str = "DQN", env=None):
    """Load a previously trained agent from disk.

    Parameters:
        path: Path to the saved model file (without .zip extension)
        algorithm: Algorithm used to train the model
        env: Optional environment to attach to the model

    Returns:
        The loaded SB3 model instance
    """
    AlgoClass = _get_algo_class(algorithm)
    return AlgoClass.load(path, env=env)


def evaluate_agent(
    model,
    env,
    n_episodes: int = 5,
    deterministic: bool = True,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Evaluate a trained agent on an environment.

    Parameters:
        model: Trained SB3 model
        env: ForgeEnv instance
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions (True for evaluation)
        verbose: Print per-episode results

    Returns:
        Dict with evaluation results:
          - episode_rewards: list of total rewards per episode
          - episode_lengths: list of steps per episode
          - final_equities: list of final equity per episode
          - mean_reward: mean total reward
          - std_reward: std of total rewards
          - mean_equity: mean final equity
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    final_equities: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_eq = float(info.get("equity", 0.0) or 0.0)
        final_equities.append(final_eq)

        if verbose >= 1:
            print(
                f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.4f}, "
                f"steps={steps}, equity={final_eq:.2f}"
            )

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_equities": final_equities,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_equity": float(np.mean(final_equities)),
        "mean_steps": float(np.mean(episode_lengths)),
    }

    if verbose >= 1:
        print(f"\nEvaluation ({n_episodes} episodes):")
        print(
            f"  Mean reward: {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}"
        )
        print(f"  Mean equity: {results['mean_equity']:.2f}")
        print(f"  Mean steps:  {results['mean_steps']:.0f}")

    return results


def train_multi_seed(
    env_fn,
    algorithm: str = "DQN",
    seeds: Optional[List[int]] = None,
    total_timesteps: int = 100_000,
    save_dir: Optional[str] = None,
    verbose: int = 1,
    **train_kwargs,
) -> List[Dict[str, Any]]:
    """Train the same agent across multiple seeds for stability evaluation.

    This is a thesis requirement: multi-seed evaluation to assess RL stability.

    Parameters:
        env_fn: Callable that returns a fresh ForgeEnv instance
        algorithm: RL algorithm name
        seeds: List of random seeds (default [42, 123, 456, 789, 1337])
        total_timesteps: Steps per seed
        save_dir: Directory to save models (one per seed)
        verbose: Verbosity level
        **train_kwargs: Extra args passed to train_agent

    Returns:
        List of dicts with {seed, model, evaluation} for each seed
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1337]

    results: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds):
        if verbose >= 1:
            print(f"\n{'=' * 60}")
            print(f"Training seed {seed} ({i + 1}/{len(seeds)})")
            print(f"{'=' * 60}")

        env = env_fn()
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{algorithm.lower()}_seed{seed}")

        model = train_agent(
            env=env,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            seed=seed,
            save_path=save_path,
            verbose=verbose,
            **train_kwargs,
        )

        # Evaluate on a fresh env
        eval_env = env_fn()
        eval_results = evaluate_agent(model, eval_env, n_episodes=3, verbose=verbose)
        eval_env.close()

        results.append(
            {
                "seed": seed,
                "model": model,
                "evaluation": eval_results,
                "save_path": save_path,
            }
        )

        env.close()

    if verbose >= 1:
        print(f"\n{'=' * 60}")
        print("Multi-seed summary:")
        rewards = [r["evaluation"]["mean_reward"] for r in results]
        equities = [r["evaluation"]["mean_equity"] for r in results]
        print(f"  Reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
        print(f"  Equity: {np.mean(equities):.2f} +/- {np.std(equities):.2f}")
        print(f"{'=' * 60}")

    return results
