import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from stable_baselines3.common.logger import Logger
# from einops import rearrange # Replaced with jnp.reshape or implicit broadcasting
from stable_baselines3.common.utils import update_learning_rate # Can be adapted or replaced
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
from stable_baselines3.common.utils import constant_fn # Kept for learning rate schedule
from stable_baselines3.common.type_aliases import ReplayBufferSamples as SB3ReplayBufferSamples # Rename to avoid conflict

# Assume chex is installed for type hinting PRNGKey, otherwise use jax.random.PRNGKey
try:
    import chex
    PRNGKey = chex.PRNGKey
except ImportError:
    # Define PRNGKey based on JAX version if chex is not available
    if hasattr(jax.random, "PRNGKey"): # Older JAX versions
        PRNGKey = jax.random.PRNGKey
    elif hasattr(jax, "Array"): # Newer JAX versions (keys are just Arrays)
         PRNGKey = jax.Array
    else:
         PRNGKey = Any # Fallback type hint

XCLIP_EMBEDDING_DIM = 768

# --- JAX Replay Buffer Samples NamedTuple ---
class RewardModelSamplesJax(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    rewards: jnp.ndarray # These are VLM returns in this context
    task_rewards: jnp.ndarray
    clip_embeddings: jnp.ndarray

# --- Replay Buffer (Modified to yield JAX arrays) ---
class PerStepReplayBufferJax(ReplayBuffer):
    def __init__(self, *args, video_length=64, n_views=1, **kwargs):
        # Remove device argument if not strictly needed for SB3 ReplayBuffer base
        original_device = kwargs.pop("device", "cpu")
        super().__init__(*args, **kwargs)
        # Ensure dimensions match buffer_size and n_envs
        self.task_rewards = np.zeros((self.buffer_size, self.n_envs, video_length), dtype=np.float32)
        self.clip_embeddings = np.zeros((self.buffer_size, self.n_envs, n_views * XCLIP_EMBEDDING_DIM), dtype=np.float32)
        # Add pred_vlm_rewards, mirroring the agent's buffer expectation for relabeling
        # Although this buffer is for the reward model, having this helps consistency?
        # Or maybe it's better added only to the *agent's* buffer. Let's omit it here for now.
        # self.pred_vlm_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.video_length = video_length
        self._original_device_info = original_device
        print(f"PerStepReplayBufferJax initialized. Data stored as NumPy, sampled as JAX arrays.")
        print(f"Original 'device' arg (for potential SB3 use): {self._original_device_info}")

    def add(
        self,
        obs: np.ndarray, # Input type hint changed to np.ndarray
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Ensure infos structure is as expected
        if not infos or "task_rewards" not in infos[0] or "clip_embedding" not in infos[0]:
             # Handle cases where info might be missing (e.g., during reset)
             # Pad with zeros or handle appropriately based on environment specifics
             # For now, assume infos[0] exists and has the keys during normal steps
             # Need to handle the case where n_envs > 1 and infos might be a list
             # Assuming infos[0] corresponds to the first env or is representative
             # This might need adjustment based on how SB3 handles multi-env infos
             if infos and isinstance(infos, list) and len(infos) > 0:
                 # Safely access keys, default to zeros if missing? Requires knowing the shape.
                 # This simplification assumes infos[0] is always valid. Robust code needs checks.
                 task_rewards_info = infos[0].get("task_rewards", np.zeros(self.task_rewards.shape[2])) # Shape (video_length,)
                 clip_embedding_info = infos[0].get("clip_embedding", np.zeros(self.clip_embeddings.shape[2])) # Shape (n_views * embed_dim,)
                 
                 # Assign potentially across multiple envs if `add` is called per env step
                 # The base class handles the multi-env logic, we just need to store the data
                 # The indexing [self.pos] suggests it handles one env transition at a time?
                 # Let's assume the base class handles broadcasting `reward`, `done` etc. from (n_envs,)
                 # We might need to replicate this for custom fields if `add` gets called once for all envs.
                 # If `add` is called per env, this is fine. If called once for all envs, needs adjustment:
                 # self.task_rewards[self.pos, env_idx] = infos[env_idx]["task_rewards"]
                 # For now, assume infos[0] is sufficient or base class handles splitting.
                 self.task_rewards[self.pos, 0] = task_rewards_info # Assumes n_envs=1 or broadcasting
                 self.clip_embeddings[self.pos, 0] = clip_embedding_info # Assumes n_envs=1 or broadcasting

             # Original simplified logic:
             # self.task_rewards[self.pos] = infos[0]["task_rewards"] # This likely assumes n_envs=1
             # self.clip_embeddings[self.pos] = infos[0]["clip_embedding"]
             # Let's refine assuming add might receive inputs for all envs:
             # reward, done shapes are typically (n_envs,)
             # obs, next_obs shapes are (n_envs, obs_dim * video_length)
             # Assuming infos is a list of dicts, one per env:
             for i in range(self.n_envs):
                 if i < len(infos): # Check if info exists for this env
                    self.task_rewards[self.pos, i] = infos[i].get("task_rewards", np.zeros(self.video_length))
                    self.clip_embeddings[self.pos, i] = infos[i].get("clip_embedding", np.zeros(self.clip_embeddings.shape[-1]))
                 else: # Handle cases where infos might be shorter than n_envs
                    self.task_rewards[self.pos, i] = np.zeros(self.video_length)
                    self.clip_embeddings[self.pos, i] = np.zeros(self.clip_embeddings.shape[-1])

        super().add(obs, next_obs, action, reward, done, infos)

    # Override _get_samples to return JAX arrays and handle video_length reshape
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[gym.Env] = None) -> RewardModelSamplesJax:
        # Sample randomly the env idx for each sample in the batch
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # --- Get data from buffer using batch and env indices ---
        if self.optimize_memory_usage:
            # Load observations (+1) and time-reverse (most probably)
            next_obs_np = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs_np = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs_np = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        actions_np = self.actions[batch_inds, env_indices, :] # Actions might not be normalized by default

        # --- Reshape based on video_length ---
        # Get dimensions assuming flattened sequences in the buffer
        obs_flat_dim = obs_np.shape[-1]
        action_flat_dim = actions_np.shape[-1]

        if obs_flat_dim % self.video_length != 0:
             raise ValueError(f"Observation dimension {obs_flat_dim} not divisible by video_length {self.video_length}")
        if action_flat_dim % self.video_length != 0:
             raise ValueError(f"Action dimension {action_flat_dim} not divisible by video_length {self.video_length}")

        obs_dim = obs_flat_dim // self.video_length
        action_dim = action_flat_dim // self.video_length
        batch_size = len(batch_inds)

        # Reshape using NumPy first
        obs_np = obs_np.reshape(batch_size, self.video_length, obs_dim)
        actions_np = actions_np.reshape(batch_size, self.video_length, action_dim)
        next_obs_np = next_obs_np.reshape(batch_size, self.video_length, obs_dim)

        # --- Get other data ---
        # Dones should be 1.0 if the episode terminates naturally, 0.0 otherwise
        # Timeout signals are separate in SB3 v2+ (self.timeouts)
        dones_np = self.dones[batch_inds, env_indices]
        timeouts_np = self.timeouts[batch_inds, env_indices]
        # Typically, done signal used in RL is 1 if not timeout, 0 otherwise
        # dones = dones * (1 - timeouts) -> ensures dones due to timeout are treated as 0 for value learning
        dones_np = (dones_np * (1 - timeouts_np)).reshape(-1, 1) # Shape (batch_size, 1)

        # Rewards are VLM returns in this context? Assuming agent buffer uses VLM return as 'reward'
        rewards_np = self.rewards[batch_inds, env_indices].reshape(-1, 1) # Shape (batch_size, 1)

        # Fetch custom data
        task_rewards_np = self.task_rewards[batch_inds, env_indices] # Shape (batch_size, video_length)
        clip_embeddings_np = self.clip_embeddings[batch_inds, env_indices] # Shape (batch_size, n_views * embed_dim)

        # --- Convert NumPy arrays to JAX arrays ---
        # Use jax.device_put for potential async dispatch and device placement
        data = (
            jax.device_put(obs_np),
            jax.device_put(actions_np),
            jax.device_put(next_obs_np),
            jax.device_put(dones_np),
            jax.device_put(rewards_np),
            jax.device_put(task_rewards_np),
            jax.device_put(clip_embeddings_np)
        )

        return RewardModelSamplesJax(*data)

# --- Reward Model Dataset (Holds JAX arrays) ---
class RewardModelDatasetJax(NamedTuple):
    train_obs: jnp.ndarray
    train_actions: jnp.ndarray
    train_vlm_returns: jnp.ndarray
    train_task_rewards: jnp.ndarray
    train_clip_embeddings: jnp.ndarray
    validate_obs: jnp.ndarray
    validate_actions: jnp.ndarray
    validate_vlm_returns: jnp.ndarray
    validate_task_rewards: jnp.ndarray
    validate_clip_embeddings: jnp.ndarray

    def _get_samples(self, batch_inds: jnp.ndarray, split: str) -> Tuple[jnp.ndarray, ...]:
        if split == "train":
            obs, actions, vlm_returns, task_rewards, clip_embeddings = (
                self.train_obs, self.train_actions, self.train_vlm_returns,
                self.train_task_rewards, self.train_clip_embeddings
            )
        elif split == "validate":
            obs, actions, vlm_returns, task_rewards, clip_embeddings = (
                self.validate_obs, self.validate_actions, self.validate_vlm_returns,
                self.validate_task_rewards, self.validate_clip_embeddings
            )
        else:
            raise ValueError(f"Invalid split name: {split}")

        # Use JAX indexing
        return (
            obs[batch_inds], actions[batch_inds], vlm_returns[batch_inds],
            task_rewards[batch_inds], clip_embeddings[batch_inds]
        )

# --- Train/Validation Split (Operates on JAX arrays) ---
def train_valid_split_jax(replay_buffer: PerStepReplayBufferJax,
                          key: PRNGKey,
                          ratio: float = 0.05,
                          env: Optional[gym.Env] = None) -> RewardModelDatasetJax:
    buffer_current_size = replay_buffer.size() # Use the method to get current size correctly
    if buffer_current_size == 0:
         raise ValueError("Cannot split an empty replay buffer.")

    if replay_buffer.full:
        buffer_limit = replay_buffer.buffer_size
        inds = np.arange(buffer_limit)
    else:
        buffer_limit = replay_buffer.pos
        inds = np.arange(buffer_limit) # Only sample up to current position

    # Sample all available data from the buffer using the modified _get_samples
    # This returns a NamedTuple containing JAX arrays
    replay_samples = replay_buffer._get_samples(inds, env)

    n_samples = len(inds)
    split_idx = int(ratio * n_samples)
    # Ensure validation set is not empty if ratio is very small but > 0
    if split_idx == 0 and ratio > 0 and n_samples > 0:
        split_idx = 1
    # Ensure training set is not empty
    if split_idx >= n_samples:
        split_idx = max(0, n_samples - 1) # Keep at least one sample for training if possible

    # Shuffle indices using JAX random functions
    key, shuffle_key = jax.random.split(key)
    shuffled_indices = jax.random.permutation(shuffle_key, jnp.arange(n_samples))

    # Apply shuffled indices to all JAX arrays in replay_samples
    shuffled_obs = replay_samples.observations[shuffled_indices]
    shuffled_actions = replay_samples.actions[shuffled_indices]
    shuffled_vlm_returns = replay_samples.rewards[shuffled_indices] # Assuming 'rewards' field holds VLM returns
    shuffled_task_rewards = replay_samples.task_rewards[shuffled_indices]
    shuffled_clip_embeddings = replay_samples.clip_embeddings[shuffled_indices]

    # Split into training and validation sets
    train_inds = shuffled_indices[:-split_idx]
    validate_inds = shuffled_indices[-split_idx:]

    return RewardModelDatasetJax(
        train_obs=shuffled_obs[:-split_idx],
        train_actions=shuffled_actions[:-split_idx],
        train_vlm_returns=shuffled_vlm_returns[:-split_idx],
        train_task_rewards=shuffled_task_rewards[:-split_idx],
        train_clip_embeddings=shuffled_clip_embeddings[:-split_idx],
        validate_obs=shuffled_obs[-split_idx:],
        validate_actions=shuffled_actions[-split_idx:],
        validate_vlm_returns=shuffled_vlm_returns[-split_idx:],
        validate_task_rewards=shuffled_task_rewards[-split_idx:],
        validate_clip_embeddings=shuffled_clip_embeddings[-split_idx:]
    )


# --- Temporal Augmentation Mask (JAX version) ---
def temporal_augmented_return_mask_jax(scores: jnp.ndarray, key: PRNGKey, min_length_ratio: float = 0.75) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    batch_size, seq_length = scores.shape[:2]

    key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)

    # Generate sub-sequence lengths
    min_len = max(1, int(seq_length * min_length_ratio))
    upper_bound = seq_length + 1
    if min_len > seq_length:
         sub_seq_lens = jnp.full((batch_size,), seq_length, dtype=jnp.int32)
    else:
         sub_seq_lens = jax.random.randint(subkey1, (batch_size,), min_len, upper_bound, dtype=jnp.int32)

    # Generate random start/end indices for mask1
    # Need length >= 2 for valid start/end pair (start < end)
    valid_start_len1 = jnp.maximum(2, sub_seq_lens)
    start_indices1 = jax.random.randint(subkey2, (batch_size,), 0, valid_start_len1 - 1)
    end_lower_bound1 = start_indices1 + 1
    end_upper_bound1 = sub_seq_lens # End index is exclusive in slicing, so upper bound is length
    # Ensure upper bound is strictly greater than lower bound for randint
    valid_end_upper1 = jnp.maximum(end_lower_bound1 + 1, end_upper_bound1 + 1) # +1 because randint is exclusive
    end_indices1 = jax.random.randint(subkey3, (batch_size,), end_lower_bound1, valid_end_upper1)

    # Generate random start/end indices for mask2
    valid_start_len2 = jnp.maximum(2, sub_seq_lens)
    start_indices2 = jax.random.randint(subkey4, (batch_size,), 0, valid_start_len2 - 1)
    end_lower_bound2 = start_indices2 + 1
    end_upper_bound2 = sub_seq_lens
    valid_end_upper2 = jnp.maximum(end_lower_bound2 + 1, end_upper_bound2 + 1)
    end_indices2 = jax.random.randint(subkey5, (batch_size,), end_lower_bound2, valid_end_upper2)

    # Create masks using jnp.arange and broadcasting
    time_indices = jnp.arange(seq_length)

    # Expand dims for broadcasting: (batch_size, 1) vs (seq_length,)
    mask1 = (time_indices >= start_indices1[:, None]) & (time_indices < end_indices1[:, None])
    mask2 = (time_indices >= start_indices2[:, None]) & (time_indices < end_indices2[:, None])

    # Ensure masks have the same number of dimensions as scores for broadcasting
    while mask1.ndim < scores.ndim:
        mask1 = jnp.expand_dims(mask1, axis=-1)
        mask2 = jnp.expand_dims(mask2, axis=-1)

    return key, mask1.astype(scores.dtype), mask2.astype(scores.dtype)


# --- Reward Model (Flax version) ---
class RewardModelFlax(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, return_embedding: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        # Input x can be (..., input_dim)
        # Flax Dense operates on the last axis.
        embedding = nn.Dense(features=512, name="dense1")(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(features=512, name="dense2")(embedding) # Backbone output

        # Normalize embedding
        embedding_norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        safe_norm = jnp.where(embedding_norm == 0, 1.0, embedding_norm)
        normalized_embedding = embedding / safe_norm

        # Predictor parameter
        predictor = self.param('predictor', nn.initializers.kaiming_uniform(), (1, 512))

        # Normalize predictor
        predictor_norm = jnp.linalg.norm(predictor, axis=-1, keepdims=True)
        safe_predictor_norm = jnp.where(predictor_norm == 0, 1.0, predictor_norm)
        normalized_predictor = predictor / safe_predictor_norm

        # Compute dot product (sum reduction over the feature dimension)
        # Broadcasting handles predictor shape (1, 512) with embedding shape (..., 512)
        output = jnp.sum(normalized_embedding * normalized_predictor, axis=-1, keepdims=True)

        if return_embedding:
            # Return the normalized embedding
            return output, normalized_embedding
        else:
            return output

# --- Main Class (JAX Version) ---
class PerStepRankingBased2Jax:
    def __init__(self, agent, key: PRNGKey):
        self.agent = agent # Assumes agent has necessary attributes (logger, env, spaces, buffer_size, etc.)
        self.video_length = agent.video_length
        self.lr_schedule = agent.reward_lr_schedule if hasattr(agent, 'reward_lr_schedule') else constant_fn(5e-4)
        self.alignment_loss_weight = agent.alignment_loss_weight
        self.temperature = agent.temperature
        self.n_top = agent.n_top
        self.reward_scale = agent.vlm_reward_scale
        self.reward_range = (-5, 5) # TODO: Make configurable?
        self.patience = 5 # TODO: Make configurable?
        self.n_min_updates = 10_000 # TODO: Make configurable?
        self.n_update_per_epoch = 50 # TODO: Make configurable?
        self.weight_decay = 1e-3 # TODO: Make configurable?

        self.key, model_key, buffer_key, train_key = jax.random.split(key, 4)
        self.train_key = train_key # Store key for training randomness

        # --- Determine observation and action dimensions ---
        # Use unwrapped spaces if available, otherwise assume direct shape access
        obs_space = agent.observation_space.unwrapped if hasattr(agent.observation_space, 'unwrapped') else agent.observation_space
        action_space = agent.action_space.unwrapped if hasattr(agent.action_space, 'unwrapped') else agent.action_space

        if isinstance(obs_space, gym.spaces.Dict):
             # Handle Dict observation space if needed
             raise NotImplementedError("Dict observation space not directly handled for Reward Model input_dim yet.")
        else:
             obs_dim = obs_space.shape[0]

        action_dim = action_space.shape[0]

        # --- Create Proxy Spaces for the Reward Model Buffer ---
        # This buffer stores flattened sequences
        proxy_obs_space = gym.spaces.Box(
            low=np.repeat(obs_space.low, self.video_length),
            high=np.repeat(obs_space.high, self.video_length),
            shape=(obs_dim * self.video_length,),
            dtype=obs_space.dtype)
        proxy_action_space = gym.spaces.Box(
            low=np.repeat(action_space.low, self.video_length),
            high=np.repeat(action_space.high, self.video_length),
            shape=(action_dim * self.video_length,),
            dtype=action_space.dtype)

        # --- Determine n_views ---
        # Safely get n_views from agent's environment configuration
        n_views = 1 # Default
        if hasattr(agent.env, 'video_sampling_configs') and isinstance(agent.env.video_sampling_configs, dict):
             n_views = agent.env.video_sampling_configs.get("n_views", 1)
        elif hasattr(agent, 'env_kwargs') and isinstance(agent.env_kwargs, dict): # Check common patterns
             video_configs = agent.env_kwargs.get('video_sampling_configs', {})
             n_views = video_configs.get("n_views", 1)

        # --- Initialize Reward Model Replay Buffer ---
        self.replay_buffer = PerStepReplayBufferJax(
            agent.reward_learning_buffer_size,
            proxy_obs_space,
            proxy_action_space,
            video_length=self.video_length,
            n_envs=1, # Reward model buffer typically processes transitions independently (n_envs=1 concept)
            n_views=n_views,
            handle_timeout_termination=False # Default SB3 buffer behavior
            )

        # --- Initialize Model and Optimizer ---
        self.model = RewardModelFlax(input_dim=obs_dim)

        # Create dummy input to initialize parameters (shape matches model's expected *step* input)
        dummy_obs_for_init = jnp.zeros((1, obs_dim), dtype=obs_space.dtype)
        self.params = self.model.init(model_key, dummy_obs_for_init)['params']

        initial_lr = self.lr_schedule(1.0) # Get initial learning rate

        # Use inject_hyperparams for dynamic LR updates during training
        self.optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=initial_lr,
            weight_decay=self.weight_decay
        )
        self.opt_state = self.optimizer.init(self.params)

        # --- Internal State for Reference Samples ---
        self.reference_rewards: Optional[jnp.ndarray] = None
        self.reference_vlm_returns: Optional[jnp.ndarray] = None
        self.reference_obs: Optional[jnp.ndarray] = None
        self.reference_task_rewards: Optional[jnp.ndarray] = None
        self.reference_rewards_flattened: Optional[jnp.ndarray] = None
        self.reference_task_rewards_flattened: Optional[jnp.ndarray] = None # Check if needed

        self.n_updates: int = 0
        self.trained: bool = False

    # --- Modified predict method ---
    def predict(self, obs: np.ndarray, action: Optional[np.ndarray] = None, task_reward: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predicts rewards. Input/Output are NumPy arrays for interface compatibility.
        Can now handle obs input with shape (batch_size, obs_dim) or (batch_size, n_envs, obs_dim).
        """
        # 1. Convert input NumPy arrays to JAX arrays
        obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
        original_shape = obs_jnp.shape
        # Determine input dimensions (D)
        obs_dim = original_shape[-1]
        # Determine total number of steps N (either B or B*E)
        num_steps = int(np.prod(original_shape[:-1])) # Use int()

        # 2. Apply model
        # Assuming model processes the last dimension, output shape will match input shape except last dim -> 1
        pred_reward_logits = self.model.apply({'params': self.params}, obs_jnp) # Shape (..., 1)

        # 3. Sample reference rewards
        # Sample one reference for each step in the input
        self.key, ref_key = jax.random.split(self.key)
        # sample_reference expects batch_size as total number of steps N
        ref_reward_logits_flat = self.sample_reference(ref_key, num_steps) # Shape (N, 1)

        # Reshape reference to match model output shape for broadcasting
        # Target shape: pred_reward_logits.shape
        ref_reward_logits = ref_reward_logits_flat.reshape(pred_reward_logits.shape)

        # 4. Apply output activation and scaling
        normalized_diff = (pred_reward_logits - ref_reward_logits) / self.temperature
        pred_reward = jax.nn.log_sigmoid(normalized_diff)
        pred_reward = self.reward_scale * pred_reward

        # 5. Clip rewards
        pred_reward = jnp.clip(pred_reward, self.reward_range[0], self.reward_range[1])

        # 6. Squeeze last dim and convert back to NumPy
        # Output shape should match input structure before the last dimension
        if pred_reward.shape[-1] == 1:
             pred_reward = pred_reward.squeeze(axis=-1)

        # Ensure result is on CPU before returning NumPy array
        pred_reward_np = np.array(jax.device_get(pred_reward))

        return pred_reward_np # Shape (...) e.g., (B, E) or (B,)

    # --- Loss Computation (Static methods) ---
    @staticmethod
    def _compute_loss(params, model_apply_fn, temperature, alignment_loss_weight,
                      obs1, vlm_returns1, clip_embeddings1,
                      obs2, vlm_returns2, clip_embeddings2):

        # pred_rewards shape: (batch, video_length, 1)
        # embeddings shape: (batch, video_length, embedding_dim)
        pred_rewards1, embeddings1 = model_apply_fn({'params': params}, obs1, return_embedding=True)
        pred_rewards2, embeddings2 = model_apply_fn({'params': params}, obs2, return_embedding=True)

        # Mean predicted reward over video length dimension (axis=1)
        pred_returns1 = pred_rewards1.mean(axis=1) # Shape: (batch, 1)
        pred_returns2 = pred_rewards2.mean(axis=1) # Shape: (batch, 1)

        # Target based on VLM returns difference (sigmoid for soft preference)
        # Scaling factor 100 from original code might cause saturation, consider adjusting.
        target = jax.nn.sigmoid((vlm_returns1 - vlm_returns2) * 100)

        # Ranking Loss (Binary Cross-Entropy with Logits)
        logits_diff = (pred_returns1 - pred_returns2) / temperature
        # Use optax's implementation for numerical stability
        ranking_loss = optax.sigmoid_binary_cross_entropy(logits=logits_diff, labels=target)
        ranking_loss = jnp.mean(ranking_loss) # Average over batch

        # Alignment Loss (Optional)
        if alignment_loss_weight > 0:
            # Mean embedding over video length
            embeddings1_mean = embeddings1.mean(axis=1) # Shape: (batch, embedding_dim)
            embeddings2_mean = embeddings2.mean(axis=1) # Shape: (batch, embedding_dim)
            # Assumes embeddings from model are already normalized

            # Cosine similarity for embeddings (sum of product for normalized vectors)
            embedding_sim = jnp.sum(embeddings1_mean * embeddings2_mean, axis=-1, keepdims=True) # Shape: (batch, 1)

            # Reshape and compute CLIP similarity
            batch_size_clip = clip_embeddings1.shape[0] # Get batch size dynamically
            n_views = clip_embeddings1.shape[-1] // XCLIP_EMBEDDING_DIM
            clip_embeddings1_reshaped = clip_embeddings1.reshape(batch_size_clip, n_views, XCLIP_EMBEDDING_DIM)
            clip_embeddings2_reshaped = clip_embeddings2.reshape(batch_size_clip, n_views, XCLIP_EMBEDDING_DIM)
            # Assume CLIP embeddings are pre-normalized

            # Clip similarity: sum over embedding dim, mean over views
            clip_sim = jnp.sum(clip_embeddings1_reshaped * clip_embeddings2_reshaped, axis=-1).mean(axis=-1, keepdims=True) # Shape: (batch, 1)

            # Alignment Loss (L1)
            alignment_loss = jnp.mean(jnp.abs(embedding_sim - clip_sim))
        else:
            alignment_loss = jnp.array(0.0) # Use jnp.array for consistency

        total_loss = ranking_loss + alignment_loss_weight * alignment_loss
        return total_loss, ranking_loss, alignment_loss

    @staticmethod
    def _compute_loss_augmented(params, model_apply_fn, temperature, alignment_loss_weight, key,
                                obs1, vlm_returns1, clip_embeddings1,
                                obs2, vlm_returns2, clip_embeddings2):
        key, subkey = jax.random.split(key)

        # Get predictions and embeddings
        pred_rewards1, embeddings1 = model_apply_fn({'params': params}, obs1, return_embedding=True)
        pred_rewards2, embeddings2 = model_apply_fn({'params': params}, obs2, return_embedding=True)

        # Generate temporal masks
        key, mask1, mask2 = temporal_augmented_return_mask_jax(pred_rewards1, subkey) # Use pred_rewards1 shape guide

        # Apply masks and SUM over time dimension (axis=1) -> Represents sum of rewards in sub-segment
        pred_returns1 = (pred_rewards1 * mask1).sum(axis=1) # Shape: (batch, 1)
        pred_returns2 = (pred_rewards2 * mask2).sum(axis=1) # Shape: (batch, 1)
        # NOTE: Original Torch code uses sum. Consider if mean is more appropriate depending on task.

        # Target and Ranking Loss (same logic as non-augmented, using summed returns)
        target = jax.nn.sigmoid((vlm_returns1 - vlm_returns2) * 100)
        logits_diff = (pred_returns1 - pred_returns2) / temperature
        ranking_loss = optax.sigmoid_binary_cross_entropy(logits=logits_diff, labels=target)
        ranking_loss = jnp.mean(ranking_loss)

        # Alignment Loss (using sum over time embeddings)
        if alignment_loss_weight > 0:
             # SUM embeddings over video length (matching reward summation)
             embeddings1_sum = embeddings1.sum(axis=1) # Shape: (batch, embedding_dim)
             embeddings2_sum = embeddings2.sum(axis=1) # Shape: (batch, embedding_dim)
             # NOTE: Original Torch code uses sum here too. Cosine sim usually uses mean/normalized.
             # Calculating cosine similarity between summed embeddings might behave differently.
             # Let's normalize the summed embeddings before dot product for cosine similarity:
             norm1 = jnp.linalg.norm(embeddings1_sum, axis=-1, keepdims=True)
             norm2 = jnp.linalg.norm(embeddings2_sum, axis=-1, keepdims=True)
             safe_norm1 = jnp.where(norm1 == 0, 1.0, norm1)
             safe_norm2 = jnp.where(norm2 == 0, 1.0, norm2)
             embeddings1_sum_norm = embeddings1_sum / safe_norm1
             embeddings2_sum_norm = embeddings2_sum / safe_norm2
             embedding_sim = jnp.sum(embeddings1_sum_norm * embeddings2_sum_norm, axis=-1, keepdims=True) # Shape: (batch, 1)

             # Clip similarity (same as non-augmented)
             batch_size_clip = clip_embeddings1.shape[0]
             n_views = clip_embeddings1.shape[-1] // XCLIP_EMBEDDING_DIM
             clip_embeddings1_reshaped = clip_embeddings1.reshape(batch_size_clip, n_views, XCLIP_EMBEDDING_DIM)
             clip_embeddings2_reshaped = clip_embeddings2.reshape(batch_size_clip, n_views, XCLIP_EMBEDDING_DIM)
             clip_sim = jnp.sum(clip_embeddings1_reshaped * clip_embeddings2_reshaped, axis=-1).mean(axis=-1, keepdims=True) # Shape: (batch, 1)

             alignment_loss = jnp.mean(jnp.abs(embedding_sim - clip_sim))
        else:
             alignment_loss = jnp.array(0.0)

        total_loss = ranking_loss + alignment_loss_weight * alignment_loss
        # Return updated key because temporal_augmented_return_mask_jax consumes key parts
        return key, total_loss, ranking_loss, alignment_loss

    def train(self, gradient_steps: int, logger: Logger, progress_remaining: float, batch_size: int = 64):
        # This is a placeholder matching the SB3 style, but logic is in batch_train
        print("Warning: train() method called, but main training logic is in batch_train().")
        # Optionally, call batch_train here if that's the intended interface
        # self.batch_train(gradient_steps, logger, progress_remaining, batch_size)
        return

    def generate_pair_inds(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        # Generates indices for all unique pairs (i, j) where i < j. Uses NumPy.
        if n_samples < 2:
            return np.array([], dtype=int), np.array([], dtype=int)
        inds1, inds2 = np.triu_indices(n_samples, k=1)
        return inds1, inds2

    def batch_train(self, gradient_steps: Optional[int], logger: Logger, progress_remaining: float, batch_size: int = 64):
        """ Train the reward model using pairwise comparisons from its internal buffer. """
        if self.replay_buffer.size() < batch_size * 2: # Need enough for pairs and split
            if logger: logger.record("train/reward_skipped_samples", 1)
            print("Skipping reward model training: Not enough samples in its buffer.")
            return

        # --- Prepare Data ---
        self.key, data_key, train_loop_key = jax.random.split(self.key, 3)

        # Determine split ratio (ensure small validation set, e.g., 50 samples or 5%)
        buffer_size = self.replay_buffer.size()
        validation_target_size = 50
        min_ratio = 0.01
        max_ratio = 0.50

        if buffer_size > validation_target_size / min_ratio: # If buffer is large enough for target size with min ratio
             validation_ratio = validation_target_size / buffer_size
        else: # Otherwise use a default small ratio
             validation_ratio = 0.05

        validation_ratio = np.clip(validation_ratio, min_ratio, max_ratio)

        try:
            dataset = train_valid_split_jax(self.replay_buffer, data_key, ratio=validation_ratio)
        except ValueError as e:
            print(f"Skipping reward model training due to error in data split: {e}")
            if logger: logger.record("train/reward_skipped_split_error", 1)
            return

        n_train, n_validate = len(dataset.train_obs), len(dataset.validate_obs)

        if n_train < 2 or n_validate < 2:
             if logger: logger.record("train/reward_skipped_pairs", 1)
             print(f"Skipping reward model training: Not enough samples for pairs after split (Train: {n_train}, Valid: {n_validate}).")
             return

        # Generate all possible pairs for validation
        validate_inds1_np, validate_inds2_np = self.generate_pair_inds(n_validate)
        validate_inds1 = jnp.asarray(validate_inds1_np)
        validate_inds2 = jnp.asarray(validate_inds2_np)
        n_validate_pairs = len(validate_inds1)

        # --- Define Training Step ---
        current_lr = self.lr_schedule(progress_remaining)

        # Choose loss function (set use_augmented_loss based on config/needs)
        use_augmented_loss = False # Example: Disable augmentation

        if use_augmented_loss:
             loss_fn = self._compute_loss_augmented
             def loss_calculator_aux(params, key, data_batch):
                  obs1, _, vlm1, _, clip1 = dataset._get_samples(data_batch[0], "train")
                  obs2, _, vlm2, _, clip2 = dataset._get_samples(data_batch[1], "train")
                  new_key, total_loss, ranking_loss, alignment_loss = loss_fn(
                       params, self.model.apply, self.temperature, self.alignment_loss_weight, key,
                       obs1, vlm1, clip1, obs2, vlm2, clip2
                  )
                  return total_loss, (new_key, ranking_loss, alignment_loss) # Return updated key in aux
             value_and_grad_fn = jax.value_and_grad(loss_calculator_aux, has_aux=True)

        else:
             loss_fn = self._compute_loss
             def loss_calculator_no_aux(params, data_batch):
                 obs1, _, vlm1, _, clip1 = dataset._get_samples(data_batch[0], "train")
                 obs2, _, vlm2, _, clip2 = dataset._get_samples(data_batch[1], "train")
                 total_loss, ranking_loss, alignment_loss = loss_fn(
                      params, self.model.apply, self.temperature, self.alignment_loss_weight,
                      obs1, vlm1, clip1, obs2, vlm2, clip2
                 )
                 return total_loss, (ranking_loss, alignment_loss) # No key update needed
             value_and_grad_fn = jax.value_and_grad(loss_calculator_no_aux, has_aux=True)

        @jax.jit
        def train_step(params, opt_state, key, current_lr, batch_inds1, batch_inds2):
            data_batch = (batch_inds1, batch_inds2)
            grad_fn_input = (params, key, data_batch) if use_augmented_loss else (params, data_batch)

            if use_augmented_loss:
                (loss, (new_key, ranking_loss, alignment_loss)), grads = value_and_grad_fn(*grad_fn_input)
            else:
                (loss, (ranking_loss, alignment_loss)), grads = value_and_grad_fn(*grad_fn_input)
                new_key = key # Key remains unchanged if not used

            # Get updates using injected hyperparams state
            updates, new_opt_state = self.optimizer.update(
                grads,
                opt_state, # Pass the current optimizer state (which includes hyperparam state)
                params,
                learning_rate=current_lr # Pass dynamic LR as keyword argument
            )
            new_params = optax.apply_updates(params, updates)

            metrics = {
                'loss': loss,
                'ranking_loss': ranking_loss,
                'alignment_loss': alignment_loss
            }
            return new_params, new_opt_state, new_key, metrics

        # --- Validation Function (JITted) ---
        @jax.jit
        def validate_step(params, batch_inds1, batch_inds2):
            # Use the non-augmented loss function for validation
            obs1, _, vlm1, _, clip1 = dataset._get_samples(batch_inds1, "validate")
            obs2, _, vlm2, _, clip2 = dataset._get_samples(batch_inds2, "validate")

            # Compute losses without gradients
            total_loss, ranking_loss, alignment_loss = self._compute_loss(
                 params, self.model.apply, self.temperature, self.alignment_loss_weight,
                 obs1, vlm1, clip1, obs2, vlm2, clip2
            )
            return total_loss, ranking_loss, alignment_loss

        # --- Training Loop ---
        min_validate_loss = float('inf')
        n_no_improvement = 0
        best_params = self.params # Store initial params as best

        # Use stored training key, propagating it through loops/steps
        current_epoch_key = self.train_key

        max_epochs = 10000 # Set a maximum number of epochs
        print(f"Starting reward model training loop (max_epochs={max_epochs}, batch_size={batch_size})...")

        for epoch in range(max_epochs):
            current_epoch_key, val_key, train_key_epoch = jax.random.split(current_epoch_key, 3)

            # --- Validation Phase ---
            epoch_validate_losses = []
            epoch_validate_ranking_losses = []
            epoch_validate_alignment_losses = []

            n_validate_batches = (n_validate_pairs + batch_size - 1) // batch_size
            for i in range(n_validate_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_validate_pairs)
                if start_idx == end_idx: continue # Skip empty batches

                batch_v_inds1 = validate_inds1[start_idx:end_idx]
                batch_v_inds2 = validate_inds2[start_idx:end_idx]

                loss, ranking_loss, alignment_loss = validate_step(
                    self.params, batch_v_inds1, batch_v_inds2
                )
                epoch_validate_losses.append(loss)
                epoch_validate_ranking_losses.append(ranking_loss)
                epoch_validate_alignment_losses.append(alignment_loss)

            # Compute average validation losses (ensure list is not empty)
            if epoch_validate_losses:
                 avg_val_loss = jnp.mean(jnp.array(epoch_validate_losses))
                 avg_val_rank_loss = jnp.mean(jnp.array(epoch_validate_ranking_losses))
                 avg_val_align_loss = jnp.mean(jnp.array(epoch_validate_alignment_losses))
                 avg_val_loss_np = float(jax.device_get(avg_val_loss)) # Get value for comparison
            else:
                 avg_val_loss = avg_val_rank_loss = avg_val_align_loss = jnp.nan
                 avg_val_loss_np = float('inf')

            # --- Early Stopping Check ---
            if avg_val_loss_np < min_validate_loss:
                min_validate_loss = avg_val_loss_np
                n_no_improvement = 0
                best_params = self.params # Save current parameters as best
                # print(f"Epoch {epoch}: New best validation loss: {min_validate_loss:.4f}") # Optional: Log improvement
            else:
                # Only check patience after minimum updates and if validation loss is valid
                if self.n_updates >= self.n_min_updates and not np.isinf(avg_val_loss_np):
                    n_no_improvement += 1
                    if n_no_improvement > self.patience:
                        print(f"Early stopping at epoch {epoch} after {self.patience} epochs without validation loss improvement.")
                        break # Exit epoch loop

            # --- Training Phase ---
            # Generate training pairs and shuffle
            train_inds1_np, train_inds2_np = self.generate_pair_inds(n_train)
            n_train_pairs = len(train_inds1_np)
            if n_train_pairs == 0:
                print(f"Epoch {epoch}: No training pairs generated, skipping training phase.")
                continue # Skip to next epoch if no pairs

            train_key_epoch, shuffle_key = jax.random.split(train_key_epoch)
            shuffled_pair_indices = jax.random.permutation(shuffle_key, jnp.arange(n_train_pairs))

            # Determine number of pairs to use in this epoch
            num_train_steps = self.n_update_per_epoch
            pairs_to_take = min(num_train_steps * batch_size, n_train_pairs) # Don't take more pairs than available

            epoch_train_indices1 = jnp.asarray(train_inds1_np)[shuffled_pair_indices[:pairs_to_take]]
            epoch_train_indices2 = jnp.asarray(train_inds2_np)[shuffled_pair_indices[:pairs_to_take]]

            epoch_train_losses = []
            epoch_train_ranking_losses = []
            epoch_train_alignment_losses = []

            n_train_batches = (len(epoch_train_indices1) + batch_size - 1) // batch_size
            current_step_key = train_key_epoch # Use the key split for this epoch's training

            for i in range(n_train_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(epoch_train_indices1))
                if start_idx == end_idx: continue

                batch_t_inds1 = epoch_train_indices1[start_idx:end_idx]
                batch_t_inds2 = epoch_train_indices2[start_idx:end_idx]

                current_step_key, step_run_key = jax.random.split(current_step_key)

                # Pass the appropriate key to train_step
                step_input_key = step_run_key if use_augmented_loss else current_step_key

                new_params, new_opt_state, post_step_key, metrics = train_step(
                    self.params, self.opt_state, step_input_key, current_lr, batch_t_inds1, batch_t_inds2
                )

                # Update state for the next step
                self.params = new_params
                self.opt_state = new_opt_state
                current_step_key = post_step_key # Update key (relevant if augmented loss used key)
                self.n_updates += 1

                epoch_train_losses.append(metrics['loss'])
                epoch_train_ranking_losses.append(metrics['ranking_loss'])
                epoch_train_alignment_losses.append(metrics['alignment_loss'])

            # --- Logging (End of Epoch) ---
            if epoch_train_losses: # Check if any training steps were taken
                 avg_train_loss = jnp.mean(jnp.array(epoch_train_losses))
                 avg_train_rank_loss = jnp.mean(jnp.array(epoch_train_ranking_losses))
                 avg_train_align_loss = jnp.mean(jnp.array(epoch_train_alignment_losses))
            else:
                 avg_train_loss = avg_train_rank_loss = avg_train_align_loss = jnp.nan

            if logger:
                 logger.record("reward_train/epoch", epoch)
                 logger.record("reward_train/validate_loss", float(jax.device_get(avg_val_loss)))
                 logger.record("reward_train/validate_ranking_loss", float(jax.device_get(avg_val_rank_loss)))
                 logger.record("reward_train/validate_alignment_loss", float(jax.device_get(avg_val_align_loss)))
                 logger.record("reward_train/train_loss", float(jax.device_get(avg_train_loss)))
                 logger.record("reward_train/train_ranking_loss", float(jax.device_get(avg_train_rank_loss)))
                 logger.record("reward_train/train_alignment_loss", float(jax.device_get(avg_train_align_loss)))
                 logger.record("reward_train/n_updates", self.n_updates)
                 logger.record("reward_train/learning_rate", current_lr)
                 logger.record("reward_train/no_improvement_epochs", n_no_improvement)

            # Update the main training key for the next epoch
            self.train_key = current_epoch_key

        # --- Post Training ---
        self.params = best_params # Restore best parameters found during training
        self.trained = True
        print(f"Finished reward model training. Total updates: {self.n_updates}. Using parameters from best validation epoch.")
        if logger:
            logger.record("reward_train/final_best_val_loss", min_validate_loss)
            logger.record("reward_train/final_total_updates", self.n_updates)

    # --- Reference Sampling ---
    def sample_reference(self, key: PRNGKey, batch_size: int) -> jnp.ndarray:
        """ Samples reference rewards (logits) from the flattened reference set. """
        if self.reference_rewards_flattened is None or len(self.reference_rewards_flattened) == 0:
            # Return zeros if no reference rewards are available
            # Ensure dtype matches model output (float32 typically)
            return jnp.zeros((batch_size, 1), dtype=jnp.float32)
        else:
            # Use jax.random.choice to sample indices
            num_available_refs = len(self.reference_rewards_flattened)
            inds = jax.random.choice(key, num_available_refs, shape=(batch_size,), replace=True)
            # Return the sampled reference logits
            return self.reference_rewards_flattened[inds] # Shape (batch_size, 1)

    # --- Modified relabel method ---
    def relabel(self):
        """ Train the model briefly and relabel rewards in the agent's buffer.
            Matches PyTorch version by passing buffer observation slices directly to predict.
        """
        # Ensure the agent's logger is available
        logger = self.agent.logger if hasattr(self.agent, 'logger') else None
        # Ensure progress remaining is available
        progress = self.agent._current_progress_remaining if hasattr(self.agent, '_current_progress_remaining') else 0.0

        print("Starting reward model training before relabeling...")
        self.batch_train(
            gradient_steps=None, # Not used by batch_train implementation here
            logger=logger,
            progress_remaining=progress,
            batch_size=256 # TODO: Make configurable
        )
        print("Reward model training finished. Starting relabeling...")

        # --- Relabel Agent's Buffer ---
        # Check if agent replay buffer exists and has the expected structure
        if not hasattr(self.agent, 'replay_buffer') or self.agent.replay_buffer is None:
            print("Error: Agent replay buffer not found. Cannot relabel.")
            return
        replay_buffer = self.agent.replay_buffer

        # Ensure the agent's buffer has 'pred_vlm_rewards' attribute to store baseline
        # This assumes the agent's buffer is a standard SB3 buffer or compatible
        if not hasattr(replay_buffer, 'pred_vlm_rewards'):
             print("Warning: Agent's replay buffer does not have 'pred_vlm_rewards'. Adding it.")
             # Initialize with zeros matching the rewards shape
             replay_buffer.pred_vlm_rewards = np.zeros_like(replay_buffer.rewards)

        relabel_batch_size = 256 # TODO: Make configurable

        # Get necessary NumPy arrays from the agent buffer
        # Note: Modifying these arrays directly modifies the buffer's contents
        try:
             old_pred_vlm_rewards_np = replay_buffer.pred_vlm_rewards
             rewards_np = replay_buffer.rewards
             observations_np = replay_buffer.observations
             # Actions might be needed if predict uses them in the future
             # actions_np = replay_buffer.actions
        except AttributeError as e:
             print(f"Error accessing agent replay buffer attributes: {e}. Cannot relabel.")
             return

        buffer_current_size = replay_buffer.size()
        if buffer_current_size == 0:
            print("Agent buffer is empty, skipping relabeling.")
            return

        buffer_limit = replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos

        # Iterate through Agent Buffer data
        for start in range(0, buffer_limit, relabel_batch_size):
            end = min(start + relabel_batch_size, buffer_limit)
            if start == end: continue

            # --- Get data directly from Agent Buffer (NumPy) ---
            # obs_batch_np shape: (current_batch_size, n_envs, obs_dim_flat) -> SB3 stores flat obs
            # We need obs_dim from the agent's space for predict
            obs_batch_flat_np = observations_np[start:end]
            # actions_batch_flat_np = actions_np[start:end] # If needed

            # --- Reshape observations for predict ---
            # Predict expects (..., obs_dim), not flattened sequence.
            # Need n_envs and obs_dim from the agent's environment/spaces.
            current_batch_size = obs_batch_flat_np.shape[0]
            n_envs = replay_buffer.n_envs
            # Get obs_dim (careful with Dict spaces)
            obs_space = self.agent.observation_space.unwrapped if hasattr(self.agent.observation_space, 'unwrapped') else self.agent.observation_space
            if isinstance(obs_space, gym.spaces.Dict):
                 raise NotImplementedError("Relabeling with Dict observation space needs specific key handling.")
            else:
                 obs_dim = obs_space.shape[0]

            # Reshape from (B, E, D_flat) to (B, E, D) if needed.
            # SB3 Buffer stores obs as (buf_size, n_envs, obs_dim). No, it stores flat! (buf_size, n_envs, obs_dim_flat)
            # Let's re-check SB3 buffer format. `add` receives obs shape `(n_envs, obs_dim)`.
            # Base ReplayBuffer stores `self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), ...)`
            # So, `observations_np` should already be `(buf_size, n_envs, *obs_shape)`.
            # Our PerStepReplayBufferJax init used proxy spaces with flattened shapes!
            # This implies the agent's buffer might store the *original* obs_shape.
            # Let's assume agent buffer `observations_np` has shape (buf_size, n_envs, obs_dim).
            obs_batch_np = observations_np[start:end] # Shape (B, E, D)

            # --- NO Reshape needed here if agent buffer stores unflattened obs ---

            # --- Use JAX model for prediction ---
            # predict accepts (B, E, D) and returns (B, E)
            # action is currently unused, passing None
            new_rewards_batch_np = self.predict(obs_batch_np, action=None) # Expected shape: (current_batch_size, n_envs)

            # --- Compare with old predicted rewards and update ---
            old_pred_batch_np = old_pred_vlm_rewards_np[start:end] # Shape: (current_batch_size, n_envs)

            if new_rewards_batch_np.shape != old_pred_batch_np.shape:
                 print(f"Warning: Shape mismatch during relabeling at step {start}: "
                       f"new_rewards {new_rewards_batch_np.shape} vs "
                       f"old_rewards {old_pred_batch_np.shape}. Skipping update for this batch.")
                 continue # Skip this batch if shapes don't match

            diff_np = new_rewards_batch_np - old_pred_batch_np # Shape: (current_batch_size, n_envs)

            # Update Agent Buffer rewards and predicted rewards (NumPy in-place modification)
            rewards_np[start:end] += diff_np
            old_pred_vlm_rewards_np[start:end] += diff_np # Update baseline

        print(f"Agent buffer relabeling finished. Processed {buffer_limit} steps.")

        # --- Update Reference Samples (using Reward Model's Buffer) ---
        print("Updating reference reward samples using reward model buffer...")
        self.key, ref_update_key = jax.random.split(self.key)

        # Get all current samples from *this* reward model's buffer
        try:
            # Use ratio=0 to get all data (returned in 'validate' fields)
            all_samples_dataset = train_valid_split_jax(self.replay_buffer, ref_update_key, ratio=0)
        except ValueError as e:
            print(f"Skipping reference update: {e}")
            return # Stop relabel if buffer is empty

        # Extract necessary data (JAX arrays)
        sample_obs = all_samples_dataset.validate_obs # Shape (n_samples, video_length, obs_dim)
        sample_vlm_returns = all_samples_dataset.validate_vlm_returns # Shape (n_samples, 1)
        sample_task_rewards = all_samples_dataset.validate_task_rewards # Shape (n_samples, video_length)

        n_samples = sample_obs.shape[0]

        if n_samples > 0:
            # Predict rewards (logits) for these samples using the *updated* model
            # Reshape obs to (n_samples * video_length, obs_dim) for step-wise prediction
            obs_steps_jnp = sample_obs.reshape(n_samples * self.video_length, sample_obs.shape[-1])

            # JIT predict_ref_logits for efficiency if called repeatedly
            @jax.jit
            def predict_ref_logits(params, obs_steps):
                 # Apply model (get logits only)
                 pred_logits = self.model.apply({'params': params}, obs_steps)
                 return pred_logits

            pred_step_logits = predict_ref_logits(self.params, obs_steps_jnp) # Shape (n_samples * video_length, 1)

            # Reshape back to sequence: (n_samples, video_length, 1)
            current_reference_rewards = pred_step_logits.reshape(n_samples, self.video_length, 1)

            # Update the reference set
            if self.reference_rewards is None: # First time setting reference
                self.reference_obs = sample_obs
                self.reference_rewards = current_reference_rewards
                self.reference_vlm_returns = sample_vlm_returns
                self.reference_task_rewards = sample_task_rewards
            else: # Append new samples to existing reference
                self.reference_obs = jnp.concatenate([self.reference_obs, sample_obs], axis=0)
                self.reference_rewards = jnp.concatenate([self.reference_rewards, current_reference_rewards], axis=0)
                self.reference_vlm_returns = jnp.concatenate([self.reference_vlm_returns, sample_vlm_returns], axis=0)
                self.reference_task_rewards = jnp.concatenate([self.reference_task_rewards, sample_task_rewards], axis=0)

            # Keep only the top N samples based on VLM returns (higher is better)
            current_ref_size = self.reference_vlm_returns.shape[0]
            if current_ref_size > self.n_top:
                # Sort by VLM returns (descending) and take top N indices
                # Ensure vlm_returns is 1D for argsort
                sort_indices = jnp.argsort(self.reference_vlm_returns.squeeze()) # Ascending sort
                top_n_indices = sort_indices[-self.n_top:] # Indices of top N highest VLM returns

                self.reference_obs = self.reference_obs[top_n_indices]
                self.reference_rewards = self.reference_rewards[top_n_indices]
                self.reference_vlm_returns = self.reference_vlm_returns[top_n_indices]
                self.reference_task_rewards = self.reference_task_rewards[top_n_indices]

            # Update flattened versions used by sample_reference
            # reference_rewards has shape (n_ref, video_length, 1) -> flatten to (n_ref * video_length, 1)
            self.reference_rewards_flattened = self.reference_rewards.reshape(-1, 1)
            # Flatten task rewards if needed for sampling (depends on usage)
            # self.reference_task_rewards_flattened = self.reference_task_rewards.reshape(-1, 1)

            print(f"Reference set updated. Current size: {self.reference_rewards.shape[0]} (kept top {self.n_top})")
        else:
            print("No samples found in reward buffer to update reference set.")
