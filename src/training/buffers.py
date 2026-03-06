import numpy as np
from typing import Any, Dict, List, Optional, NamedTuple
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from vlm_tqc import VLMReplayBuffer

class WAReplayBuffer(VLMReplayBuffer):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    actions_weight: np.ndarray
    task_rewards: np.ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rwd_dim = 1
        self.task_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.actions_weight = np.zeros((self.buffer_size, self.n_envs, self.rwd_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if 'action_weight' in infos[0]:
            action_weight = np.array(
                [info['action_weight'] for info in infos])
            action_weight = action_weight.reshape((self.n_envs, self.rwd_dim))
            self.actions_weight[self.pos] = action_weight
        if 'task_reward' in infos[0]:
            task_reward = np.array(
                [info['task_reward'] for info in infos])
            self.task_rewards[self.pos] = task_reward
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.actions_weight[batch_inds, env_indices, :],
            self._normalize_reward(self.task_rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    actions_weight: np.ndarray
    task_rewards: np.ndarray