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

XCLIP_EMBEDDING_DIM = 768

class PerStepReplayBuffer(ReplayBuffer):
    def __init__(self, *args, video_length=64, n_views=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_rewards = np.zeros((self.buffer_size, self.n_envs, video_length))
        self.clip_embeddings = np.zeros((self.buffer_size, self.n_envs, n_views * XCLIP_EMBEDDING_DIM), dtype=np.float32) # assume using the X-CLIP model
        self.video_length = video_length
    
    def add(
        self,
        obs: np.typing.NDArray,
        next_obs: np.typing.NDArray,
        action: np.typing.NDArray,
        reward: np.typing.NDArray,
        done: np.typing.NDArray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.task_rewards[self.pos] = infos[0]["task_rewards"]
        self.clip_embeddings[self.pos] = infos[0]["clip_embedding"]
        super().add(obs, next_obs, action, reward, done, infos)
    
    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
            # Sample randomly the env idx
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
            if self.optimize_memory_usage:
                next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
            else:
                next_obs = self.next_observations[batch_inds, env_indices, :]
            obs = rearrange(
                self.observations[batch_inds, env_indices, :], 
                ("batch_size (obs_dim video_length) "
                "-> batch_size video_length obs_dim"),
                video_length=self.video_length,)
            actions = rearrange(
                self.actions[batch_inds, env_indices, :], 
                ("batch_size (action_dim video_length) "
                "-> batch_size video_length action_dim"),
                video_length=self.video_length,)
            next_obs = rearrange(
                next_obs, 
                ("batch_size (obs_dim video_length) "
                "-> batch_size video_length obs_dim"),
                video_length=self.video_length,)
            data = (
                self._normalize_obs(obs, env),
                actions,
                self._normalize_obs(next_obs, env),
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                self.rewards[batch_inds, env_indices].reshape(-1, 1),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

class RewardModelDataset(NamedTuple):
    train_obs: th.Tensor
    train_actions: th.Tensor
    train_vlm_returns: th.Tensor
    train_task_rewards: th.Tensor
    train_clip_embeddings: th.Tensor
    validate_obs: th.Tensor
    validate_actions: th.Tensor
    validate_vlm_returns: th.Tensor
    validate_task_rewards: th.Tensor
    validate_clip_embeddings: th.Tensor
    def _get_samples(self, batch_inds, split):
        if split == "train":
            obs  = self.train_obs
            actions = self.train_actions
            vlm_returns = self.train_vlm_returns
            task_rewards = self.train_task_rewards
            clip_embeddings = self.train_clip_embeddings
        else:
            obs  = self.validate_obs
            actions = self.validate_actions
            vlm_returns = self.validate_vlm_returns
            task_rewards = self.validate_task_rewards
            clip_embeddings = self.validate_clip_embeddings
            
        return obs[batch_inds], actions[batch_inds], vlm_returns[batch_inds], task_rewards[batch_inds], clip_embeddings[batch_inds]
         
def train_valid_split(replay_buffer, ratio=0.05, env=None):
    if replay_buffer.full:
        inds = np.arange(replay_buffer.buffer_size)
    else:
        inds = np.arange(replay_buffer.pos)
    replay_samples = replay_buffer._get_samples(inds, env)
    shuffle_inds = np.random.permutation(len(inds))
    obs = replay_samples.observations[shuffle_inds]
    actions = replay_samples.actions[shuffle_inds]
    vlm_returns = replay_samples.rewards[shuffle_inds]
    task_rewards = replay_buffer.task_rewards[shuffle_inds]
    task_rewards = th.from_numpy(task_rewards).to(replay_buffer.device)
    clip_embeddings = replay_buffer.clip_embeddings[shuffle_inds]
    clip_embeddings = th.from_numpy(clip_embeddings).to(replay_buffer.device)
    split = int(ratio * len(inds))

    return RewardModelDataset(obs[:-split],
            actions[:-split],
            vlm_returns[:-split],
            task_rewards[:-split],
            clip_embeddings[:-split],
            obs[-split:],
            actions[-split:],
            vlm_returns[-split:],
            task_rewards[-split:],
            clip_embeddings[-split:])

def temporal_augmented_return_mask(scores, min_length_ratio=0.75):
    # Extract batch and sequence dimensions
    batch_size, seq_length = scores.shape[:2]
    sub_seq_lens = np.random.randint(
        int(seq_length * min_length_ratio), seq_length+1, size=batch_size)
    mask1 = th.zeros_like(scores)
    for i in range(batch_size):
        start_idx = np.random.randint(0, sub_seq_lens[i] - 1)
        end_idx = np.random.randint(start_idx + 1, sub_seq_lens[i])
        mask1[i, start_idx:end_idx] = 1
    mask2 = th.zeros_like(scores)
    for i in range(batch_size):
        start_idx = np.random.randint(0, sub_seq_lens[i] - 1)
        end_idx = np.random.randint(start_idx + 1, sub_seq_lens[i])
        mask2[i, start_idx:end_idx] = 1
    return mask1, mask2

class RewardModel(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.backbone = th.nn.Sequential(
            th.nn.Linear(input_dim, 512),
            th.nn.ReLU(),
            th.nn.Linear(512, 512))
        self.predictor = th.nn.Parameter(th.empty(1, 512))
        th.nn.init.kaiming_uniform_(self.predictor)
    def forward(self, input, return_embedding=False):
        embedding = self.backbone(input)
        embedding /= embedding.clone().norm(dim=-1, keepdim=True)
        predictor = self.predictor / self.predictor.norm(dim=-1, keepdim=True)
        predictor = self.predictor.expand_as(embedding)
        output = (embedding * predictor).sum(-1,keepdim=True)
        # batch, 256 
        if return_embedding:
            return output, embedding
        else:
            return output

class PerStepRankingBased2(th.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.device = agent.device
        self.video_length = agent.video_length
        self.lr_schedule = constant_fn(5e-4)
        
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
        self.replay_buffer = PerStepReplayBuffer(
            agent.reward_learning_buffer_size, 
            proxy_obs_space,
            proxy_action_space,
            device=self.device,
            video_length=self.video_length,
            n_envs=1,
            n_views=agent.env.video_sampling_configs["n_views"])
        input_dim = agent.observation_space.shape[0]
        self.model = RewardModel(input_dim)
        self.optimizer = th.optim.AdamW(
            self.model.parameters(), self.lr_schedule(1), weight_decay=1e-3)
        self.to(self.device)
        self.loss_func = th.nn.BCEWithLogitsLoss()
        self.alignment_loss_func = th.nn.L1Loss()
        self.alignment_loss_weight = agent.alignment_loss_weight
        self.output_activation_func = th.nn.LogSigmoid()
        self.patience = 5
        self.n_min_updates = 10_000
        self.n_update_per_epoch = 50 # 100 for soft labels of pairwise, 10 for utility based
        self.reference_rewards = None
        self.n_updates = 0
        self.temperature = agent.temperature
        self.n_top = agent.n_top
        self.reward_scale = agent.vlm_reward_scale
        self.reward_range = (-5, 5)
        self.trained = False
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
        pred_reward = self.output_activation_func(
            (pred_reward - ref_reward) / self.temperature)
        #baseline = self.output_activation_func(th.zeros_like(pred_reward))
        pred_reward = self.reward_scale * pred_reward
        pred_reward = pred_reward.clip(*self.reward_range)
        pred_reward = pred_reward.squeeze().cpu().numpy()   
        return pred_reward
    
    def _compute_loss_from_samples(
        self, obs1, actions1, vlm_returns1, task_rewards1, clip_embeddings1, obs2, actions2,vlm_returns2, task_rewards2, clip_embeddings2):
        pred_rewards1, embeddings1 = self.model(obs1, return_embedding=True)
        pred_returns1 = pred_rewards1.mean(1)
        pred_rewards2, embeddings2 = self.model(obs2, return_embedding=True)
        pred_returns2 = pred_rewards2.mean(1)
        target = th.sigmoid((vlm_returns1 - vlm_returns2)* 100)
        ranking_loss = self.loss_func(
            (pred_returns1 - pred_returns2) / self.temperature, target)
        if self.alignment_loss_weight > 0:
            # alignment
            embeddings1 = embeddings1.mean(1)
            embeddings2 = embeddings2.mean(1)
            embedding_sim = (embeddings1 * embeddings2).sum(-1, keepdim=True) # batch, 1
            clip_embeddings1 = clip_embeddings1.reshape(
                (embeddings1.shape[0], -1, XCLIP_EMBEDDING_DIM))
            clip_embeddings2 = clip_embeddings2.reshape(
                (embeddings1.shape[0], -1, XCLIP_EMBEDDING_DIM))
            clip_sim = (clip_embeddings1 * clip_embeddings2).sum(-1).mean(-1, keepdim=True) # batch, 1
            alignment_loss = self.alignment_loss_func(embedding_sim, clip_sim)
        else:
            alignment_loss = th.zeros_like(ranking_loss)
        
        total_loss = ranking_loss + self.alignment_loss_weight * alignment_loss
        return  total_loss, ranking_loss, alignment_loss

    def _compute_loss_from_samples_augmented(
        self, obs1, actions1, vlm_returns1, task_rewards1, clip_embeddings1, obs2, actions2,vlm_returns2, task_rewards2, clip_embeddings2):
        pred_rewards1, embeddings1 = self.model(obs1, return_embedding=True)
        masks1, masks2 = temporal_augmented_return_mask(
            pred_rewards1)
        pred_returns1 = (pred_rewards1 * masks1).sum(1)
        pred_rewards2, embeddings2 = self.model(obs2, return_embedding=True)
        pred_returns2 = (pred_rewards2 * masks2).sum(1) 
        target = th.sigmoid((vlm_returns1 - vlm_returns2)* 100)
        ranking_loss = self.loss_func(
            (pred_returns1 - pred_returns2) / self.temperature, target)
        
        # alignment
        embeddings1 = embeddings1.sum(1)
        embeddings2 = embeddings2.sum(1)
        embedding_sim = (embeddings1 * embeddings2).sum(-1, keepdim=True) # batch, 1
        clip_embeddings1 = clip_embeddings1.reshape(
            (embeddings1.shape[0], -1, 768))
        clip_embeddings2 = clip_embeddings2.reshape(
            (embeddings1.shape[0], -1, 768))
        clip_sim = (clip_embeddings1 * clip_embeddings2).sum(-1).mean(-1, keepdim=True) # batch, 1
        alignment_loss = self.alignment_loss_func(embedding_sim, clip_sim)
        
        total_loss = ranking_loss + self.alignment_loss_weight * alignment_loss
        return  total_loss, ranking_loss, alignment_loss
    
    def train(self,
              gradient_steps: int,
              logger: Logger,
              progress_remaining: float,
              batch_size: int=64):     
        return

    def generate_pair_inds(self, n_validate):
        inds1, inds2 = [], []
        for i in range(n_validate):
            for j in range(i+1, n_validate):
                inds1.append(i)
                inds2.append(j)
        return np.array(inds1), np.array(inds2)
                
    def batch_train(self,
              gradient_steps: int,
              logger: Logger,
              progress_remaining: float,
              batch_size: int=64):
        if self.replay_buffer.size() < batch_size:
            return
        if self.reference_rewards is None:
            return 
        self.model.train()
        update_learning_rate(
            self.optimizer, self.lr_schedule(progress_remaining))
        
        if self.replay_buffer.pos < 1000 and (not self.replay_buffer.full):
            dataset = train_valid_split(self.replay_buffer,)
        else:
            dataset = train_valid_split(
                self.replay_buffer, ratio=50 / self.replay_buffer.size())
        n_train, n_validate = len(dataset.train_obs), len(dataset.validate_obs)
        train_inds1, train_inds2 = self.generate_pair_inds(n_train)
        validate_inds1, validate_inds2 = self.generate_pair_inds(n_validate)
        min_validate_loss = np.inf
        n_no_improvement = 0
        for epoch in range(10000):
            epoch_validate_loss = []
            epoch_validate_ranking_loss = []
            epoch_validate_alignment_loss = []
            with th.no_grad():
                for batch in range(0, len(validate_inds1), batch_size):
                    loss, ranking_loss, alignment_loss  = self._compute_loss_from_samples(
                    *dataset._get_samples(
                        validate_inds1[batch: batch+batch_size], "validate"),
                    *dataset._get_samples(
                        validate_inds2[batch: batch+batch_size], "validate"))
                    epoch_validate_loss.append(loss.item())
                    epoch_validate_ranking_loss.append(ranking_loss.item())
                    epoch_validate_alignment_loss.append(alignment_loss.item())
            epoch_validate_loss = np.mean(epoch_validate_loss)
            epoch_validate_ranking_loss = np.mean(epoch_validate_ranking_loss)
            epoch_validate_alignment_loss = np.mean(epoch_validate_alignment_loss)
            if epoch_validate_loss < min_validate_loss:
                min_validate_loss = epoch_validate_loss
                n_no_improvement = 0
            else:
                if self.n_updates > self.n_min_updates:
                    n_no_improvement += 1
                    if n_no_improvement > self.patience:
                        print(f"Early stop at epoch {epoch}")
                        break
            epoch_train_loss = []
            epoch_train_ranking_loss = []
            epoch_train_alignment_loss = []
            
            shuffle_inds = np.random.permutation(len(train_inds1))
            inds1 = train_inds1[shuffle_inds][
                :self.n_update_per_epoch * batch_size]
            inds2 = train_inds2[shuffle_inds][
                :self.n_update_per_epoch * batch_size]
            for batch in range(0, len(inds1), batch_size):
                loss, ranking_loss, alignment_loss = self._compute_loss_from_samples(
                    *dataset._get_samples(
                            inds1[batch: batch+batch_size], "train"),
                    *dataset._get_samples(
                            inds2[batch: batch+batch_size], "train"))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.n_updates += 1
                epoch_train_loss.append(loss.item())
                epoch_train_ranking_loss.append(ranking_loss.item())
                epoch_train_alignment_loss.append(alignment_loss.item())
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_ranking_loss = np.mean(epoch_train_ranking_loss)
            epoch_train_alignment_loss = np.mean(epoch_train_alignment_loss)
        logger.record("train/reward_validate_loss", epoch_validate_loss)
        logger.record("train/reward_validate_ranking_loss", epoch_validate_ranking_loss)
        logger.record("train/reward_validate_alignment_loss", epoch_validate_alignment_loss)
        logger.record("train/reward_train_loss", epoch_train_loss)
        logger.record("train/reward_train_ranking_loss", epoch_train_ranking_loss)
        logger.record("train/reward_train_alignment_loss", epoch_train_alignment_loss)
        logger.record("train/reward_n_update", self.n_updates)
        self.model.eval()
        self.trained = True
        
    @th.inference_mode()
    def sample_reference(self, batch_size):
        if self.reference_rewards is None:
            return th.zeros((batch_size, 1), device=self.device)
        else:
            inds = np.random.choice(
                len(self.reference_rewards_flattened), size=batch_size)
            return self.reference_rewards_flattened[inds] 

    def relabel(self):
        self.batch_train(
            gradient_steps=None,
            logger=self.agent.logger,
            progress_remaining=self.agent._current_progress_remaining,
            batch_size=256)
        # relable rewards in the agent's buffer
        replay_buffer = self.agent.replay_buffer
        batch_size = 256 #self.agent.batch_size
        old_pred_vlm_rewards = self.agent.replay_buffer.pred_vlm_rewards
        if replay_buffer.full:
            end = replay_buffer.buffer_size
        else:
            end = replay_buffer.pos
        with th.no_grad():
            for start in range(0, end, batch_size):
                old_reward = old_pred_vlm_rewards[start: start + batch_size]
                obs = replay_buffer.observations[start: start + batch_size]
                action = replay_buffer.actions[start: start + batch_size]
                new_reward = self.predict(obs, action)
                diff = new_reward - old_reward
                replay_buffer.rewards[start: start + batch_size] += diff
                old_pred_vlm_rewards[start: start + batch_size] += diff
        # update reference vector
        all_samples = train_valid_split(self.replay_buffer, ratio=0,)
        n_samples = len(all_samples.validate_actions)
        sample_rewards = []
        with th.no_grad():
            for batch in range(0, n_samples, batch_size):
                obs = all_samples.validate_obs[batch: batch + batch_size]
                pred_rewards = self.model(obs)
                sample_rewards.append(pred_rewards)
            # relabel reference samples
            if not self.reference_rewards is None:
                self.referece_rewards = self.model(self.reference_obs)
        vlm_returns = all_samples.validate_vlm_returns
        if n_samples > 0:
            if len(sample_rewards) > 0:
                sample_rewards = th.cat(sample_rewards, dim=0)
            else:
                sample_rewards = sample_rewards[0]
            sample_obs = all_samples.validate_obs
            if self.reference_rewards is None:
                self.reference_rewards = sample_rewards
                self.reference_vlm_returns = vlm_returns
                self.reference_obs = sample_obs
                self.reference_task_rewards = all_samples.validate_task_rewards
            else:
                self.reference_rewards = th.cat(
                    [self.reference_rewards, sample_rewards])
                self.reference_vlm_returns = th.cat(
                    [self.reference_vlm_returns, vlm_returns])
                self.reference_obs = th.cat(
                    [self.reference_obs, sample_obs])
                self.reference_task_rewards = th.cat(
                    [self.reference_task_rewards, all_samples.validate_task_rewards])
            if self.reference_vlm_returns.shape[0] > self.n_top:    
                _, sort_ind = self.reference_vlm_returns.squeeze().sort()
                self.reference_vlm_returns = self.reference_vlm_returns[sort_ind[-self.n_top:]]
                self.reference_rewards = self.reference_rewards[-self.n_top:]
                self.reference_obs = self.reference_obs[-self.n_top:]
                self.reference_task_rewards = self.reference_task_rewards[-self.n_top:]
            self.reference_rewards_flattened = self.reference_rewards.view((
                -1, 1))
            self.reference_task_rewards_flattened = self.reference_task_rewards.view((
                -1, 1))

