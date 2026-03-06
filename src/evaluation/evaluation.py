import warnings
from typing import Any, Callable, Optional, Union, Dict, List

import gymnasium as gym
import numpy as np
import jax

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,

) -> Union[tuple[float, float], tuple[list[float], list[int]]]:

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        # Wrap non-Vec environments as DummyVecEnv
        env = DummyVecEnv([lambda: env])

    # Check whether the env is wrapped by Monitor or VecMonitor
    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with `Monitor`; results may be inaccurate.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_successes = [] # Fixed the misspelling from the original implementation

    episode_counts = np.zeros(n_envs, dtype="int")
    # Spread evaluation episodes as evenly as possible across sub-envs
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset() # Reset the environment and obtain the initial observation
    states = None # RNN states, if applicable
    episode_starts = np.ones((env.num_envs,), dtype=bool) # Track whether each env is at the beginning of an episode



    # --- Main evaluation loop ---
    while (episode_counts < episode_count_targets).any(): # Continue while any environment still needs episodes
        # Keep observations for critic input before prediction
        observations_for_critic = observations

        # Query the policy for actions
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic, # whether to use deterministic actions
        )

        # --- Step the environment ---
        new_observations, rewards, dones, infos = env.step(actions)

        # --- Standard logging and episode bookkeeping ---
        current_rewards += rewards # Accumulate episodic reward
        current_lengths += 1      # Increment episodic length
        for i in range(n_envs): # Iterate over each parallel environment
            if episode_counts[i] < episode_count_targets[i]: # Only process envs that still need episodes
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                # Safely fetch success flag; default to False when missing
                info["is_success"] = info.get("success", False)
                episode_starts[i] = done # When done=True, mark next iteration with episode_start=True

                # Execute callback when provided
                if callback is not None:
                    callback(locals(), globals())


                # --- Handle episode termination ---
                if dones[i]:
                    if is_monitor_wrapped:
                        # When Monitor-wrapped, read true returns/lengths from info
                        if "episode" in info:
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Safely extract success flag from info
                            episode_successes.append(info.get("is_success", False))
                            # Increment counts only on real episode termination
                            episode_counts[i] += 1
                    else:
                        # Without Monitor, rely on accumulated reward and length
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_successes.append(info.get("is_success", False))
                        episode_counts[i] += 1

                    # Reset per-env accumulators
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        # Update observations for the next loop iteration
        observations = new_observations

    # --- Compute and return metrics ---
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0

    # Validate reward threshold when specified
    if reward_threshold is not None:
        if not episode_rewards: # Handle the case where no episodes finished
             print("Warning: no evaluation episodes completed; cannot validate reward threshold.")
        else:
             assert mean_reward > reward_threshold, (
                 f"Mean reward {mean_reward:.2f} did not reach threshold {reward_threshold:.2f}"
             )

    # Return detailed episode metrics or summary statistics
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_successes
    return mean_reward, std_reward, episode_successes # Also return the list of successes
