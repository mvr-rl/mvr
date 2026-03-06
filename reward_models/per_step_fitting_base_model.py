from typing import Any, Dict, List
import torch as th
import numpy as np
from stable_baselines3.common.logger import Logger
from einops import rearrange
from stable_baselines3.common.utils import update_learning_rate
from typing import NamedTuple
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from stable_baselines3.common.utils import update_learning_rate, polyak_update
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.type_aliases import ReplayBufferSamples
CLIP_EMBEDDING_DIM = 1024
from .per_step_ranking_base_model2 import PerStepRankingBased2, RewardModelDataset, temporal_augmented_return_mask, train_valid_split, RewardModel, PerStepReplayBuffer

class PerStepReplayBuffer2(PerStepReplayBuffer):
    def __init__(self, *args, video_length=64, n_views=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_embeddings = np.zeros((self.buffer_size, self.n_envs, n_views * CLIP_EMBEDDING_DIM), dtype=np.float32) # assume using the X-CLIP model

class PerStepFittingBased2(PerStepRankingBased2):
    def __init__(self, agent):
        super().__init__(agent)
        self.loss_func = th.nn.MSELoss()
        self.alignment_loss_weight = 0
        proxy_obs_space = gym.spaces.Box(
            low=np.repeat(agent.observation_space.low, self.video_length),
            high=np.repeat(agent.observation_space.high, self.video_length),
            shape=(agent.observation_space.shape[0] * self.video_length,), 
            dtype=np.float32)
        proxy_action_space = gym.spaces.Box(
            low=np.repeat(agent.action_space.low, self.video_length),
            high=np.repeat(agent.action_space.high, self.video_length),
            shape=(agent.action_space.shape[0] * self.video_length,), 
            dtype=np.float32)
        self.replay_buffer = PerStepReplayBuffer2(
            agent.reward_learning_buffer_size, 
            proxy_obs_space,
            proxy_action_space,
            device=self.device,
            video_length=self.video_length,
            n_envs=1,
            n_views=agent.env.video_sampling_configs["n_views"])
    @th.inference_mode
    def predict(self, obs: np.typing.NDArray, action:np.typing.NDArray, task_reward:np.typing.NDArray=None):
        obs = th.from_numpy(obs).to(th.float32).to(self.device)
        pred_reward = self.model(obs)
        if len(pred_reward.shape) == 2:
            ref_reward = self.sample_reference(pred_reward.shape[0])
        elif len(pred_reward.shape) == 3:
            ref_reward = self.sample_reference(
                pred_reward.shape[0]*pred_reward.shape[1])
            ref_reward = ref_reward.reshape(pred_reward.shape)
        else:
            raise NotImplementedError 
        pred_reward = pred_reward
        pred_reward = pred_reward.clip(*self.reward_range)
        pred_reward = pred_reward.squeeze().cpu().numpy()   
        return pred_reward
    
    def _compute_loss_from_samples(
        self, obs1, actions1, vlm_returns1, task_rewards1, clip_embeddings1, obs2, actions2,vlm_returns2, task_rewards2, clip_embeddings2):
        pred_rewards1, embeddings1 = self.model(obs1, return_embedding=True)
        pred_returns1 = pred_rewards1.mean(1)
        pred_rewards2, embeddings2 = self.model(obs2, return_embedding=True)
        pred_returns2 = pred_rewards2.mean(1)
        ranking_loss = self.loss_func(
            pred_returns1 , vlm_returns1)+self.loss_func(
            pred_returns2, vlm_returns2)
        alignment_loss = th.zeros_like(ranking_loss)
        
        total_loss = ranking_loss + self.alignment_loss_weight * alignment_loss
        return  total_loss, ranking_loss, alignment_loss