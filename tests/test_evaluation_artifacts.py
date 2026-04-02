from __future__ import annotations

from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from evaluation.artifacts import build_model_eval_env, evaluate_rl_model_on_period


class DummyTradingEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=-1_000.0,
            high=1_000.0,
            shape=(2,),
            dtype=np.float32,
        )
        self._step_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        return np.array([10.0, -10.0], dtype=np.float32), {"equity": 100.0}

    def step(self, action):
        self._step_idx += 1
        obs = np.array([10.0 + self._step_idx, -10.0], dtype=np.float32)
        terminated = self._step_idx >= 3
        info = {
            "equity": 100.0 + float(self._step_idx),
            "is_liquidated": False,
        }
        return obs, 0.0, terminated, False, info


class DummyModel:
    def __init__(self, train_vec_norm=None):
        self._train_vec_norm = train_vec_norm

    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            return np.zeros((obs.shape[0],), dtype=np.int64), None
        return 0, None

    def get_vec_normalize_env(self):
        return self._train_vec_norm


def _make_train_vec_norm() -> VecNormalize:
    train_env = VecNormalize(
        DummyVecEnv([DummyTradingEnv]),
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    obs = train_env.reset()
    for _ in range(4):
        action = np.zeros((1,), dtype=np.int64)
        obs, _reward, dones, _infos = train_env.step(action)
        if bool(dones[0]):
            obs = train_env.reset()
    return train_env


def test_build_model_eval_env_copies_obs_stats():
    train_vec_norm = _make_train_vec_norm()
    model = DummyModel(train_vec_norm)

    eval_env = build_model_eval_env(DummyTradingEnv, model)

    assert isinstance(eval_env, VecNormalize)
    assert eval_env.training is False
    assert eval_env.obs_rms is not train_vec_norm.obs_rms
    np.testing.assert_allclose(eval_env.obs_rms.mean, train_vec_norm.obs_rms.mean)
    np.testing.assert_allclose(eval_env.obs_rms.var, train_vec_norm.obs_rms.var)

    metrics = evaluate_rl_model_on_period(
        model,
        eval_env,
        starting_cash=100.0,
        start_iso=datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        end_iso=datetime(2024, 1, 1, 3, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        timeframe="1h",
    )
    assert metrics["total_steps"] == 3
    assert metrics["equities"][-1] == 103.0
    eval_env.close()
    train_vec_norm.close()


def test_evaluate_rl_model_on_period_supports_plain_gym_env():
    model = DummyModel()
    env = DummyTradingEnv()

    metrics = evaluate_rl_model_on_period(
        model,
        env,
        starting_cash=100.0,
        start_iso="2024-01-01T00:00:00Z",
        end_iso="2024-01-01T03:00:00Z",
        timeframe="1h",
    )

    assert metrics["total_steps"] == 3
    assert metrics["equities"] == [100.0, 101.0, 102.0, 103.0]
