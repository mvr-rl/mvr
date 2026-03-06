import logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, Union
from collections import deque
import os
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from einops import rearrange
from collections import defaultdict
import numpy as np
import torch as th
import wandb
from einops import rearrange
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from src.evaluation.evaluation import evaluate_policy
# <<< import flax for type checking and potential parameter operations >>>
import flax.core
# <<< import TrainState for type checking >>>
from flax.training.train_state import TrainState


logger = logging.getLogger(__name__)

class LogTrainingStats(BaseCallback):
    def _on_training_start(self) -> None:
        reset_num_timesteps = self.locals["reset_num_timesteps"]
        if reset_num_timesteps:
            self.model.train_stats_buffer = {}
            for key in self.model.info_keys_to_print:
                self.model.train_stats_buffer[key] = deque(
                    maxlen=self.model._stats_window_size)
        return super()._on_training_start()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            for key in self.model.info_keys_to_print:
                if key in info:
                    self.model.train_stats_buffer[key].append(info[key])
        return True

class WandbCallback(SB3WandbCallback):
    def save_model(self) -> None:
        # Overload this method so that we don't upload model ckpt to wandb
        # and add a prefix to model ckpts
        save_path = os.path.join(
            self.model_save_path,
            f"{self.model.num_timesteps}_steps.zip")
        self.model.save(save_path)
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {save_path}")

class CollectClipsCallback(BaseCallback):
    def __init__(
        self,
        vlm,
        env: Union[gym.Env, VecEnv],
        verbose: int = 0,
        n_episodes_per_call=10,
        encoding_batch_size=16,
        max_clips_per_iteration=400):
        super().__init__(verbose=verbose)
        self.vlm = vlm.to("cuda")
        self.env = env
        self.n_episodes_per_call = n_episodes_per_call
        self.encoding_batch_size = encoding_batch_size
        self.max_clips_per_iteration = max_clips_per_iteration
        self.smoothing_vlm_returns = False
        self.smoothing_window_size = 5
        self.smoothing_weights = np.ones(self.smoothing_window_size) / self.smoothing_window_size

    def _on_training_start(self) -> None:
        return super()._on_training_start()

    def _on_step(self) -> bool:
        clips = self.env.collect_clips(self.max_clips_per_iteration)
        if len(clips) >= self.max_clips_per_iteration:
            clips = np.random.permutation(clips)[:self.max_clips_per_iteration]
        if len(clips) == 0:
            return True
        render_arrays = np.array([clip["render_arrays"] for clip in clips])
        video_length = render_arrays.shape[1]
        views = []
        for view_ind in range(render_arrays.shape[-1]//3):
            view_array = render_arrays[:, :, :, :, 3*view_ind: 3*(view_ind+1)]
            views.append(view_array)
        vlm_returns = []
        clip_embeddings = []
        for bid in range(
            0, len(render_arrays), self.encoding_batch_size):
            batch_vlm_returns = 0
            batch_embeddings = []
            for view_array in views:
                view_batch = view_array[bid: bid + self.encoding_batch_size]
                embedding = self.vlm.encode_stacked_image(
                    view_batch, n_stack=video_length)
                embedding = embedding.unsqueeze(1)
                view_batch_vlm_returns = self.vlm(embedding).squeeze()
                # batch_vlm_returns = batch_vlm_returns1
                batch_vlm_returns += view_batch_vlm_returns
                embedding = embedding[:,:,-1]
                embedding = embedding / embedding.norm(
                    p=2, dim=-1, keepdim=True)
                batch_embeddings.append(embedding)
            if len(batch_embeddings) > 1:
                batch_embedding = th.concat(batch_embeddings, 1)
            else:
                batch_embedding = batch_embeddings[0]
            batch_vlm_returns /= len(views)
            if len(batch_vlm_returns.shape) == 0:
                # when batch_vlm_reward is a zero-dimensional array
                batch_vlm_returns = batch_vlm_returns[None]
            vlm_returns.append(batch_vlm_returns)
            clip_embeddings.append(batch_embedding.flatten(2).cpu())
        vlm_returns = np.concatenate(vlm_returns, 0)
        clip_embeddings = np.concatenate(clip_embeddings)
        if self.smoothing_vlm_returns:
            clips_by_trajectory = defaultdict(list)
            for cid, clip in enumerate(clips):
                trajectory_id = clip["info"]["trajectory_id"]
                clips_by_trajectory[trajectory_id].append(cid)
            for _, clip_inds in clips_by_trajectory.items():
                steps = [clips[ind]["info"]["step"] for ind in clip_inds]
                sorted_inds = np.argsort(steps)
                traj_vlm_returns = [\
                    vlm_returns[
                        clip_inds[sorted_ind]] for sorted_ind in sorted_inds]
                smoothed_vlm_returns = np.convolve(
                    traj_vlm_returns, self.smoothing_weights, mode='same')
                for sorted_ind in sorted_inds:
                    clip_ind = clip_inds[sorted_ind]
                    clip = clips[clip_ind]
                    clip_info = clip["info"]
                    clip_info["task_rewards"] = clip["rewards"]
                    self.model.reward_model.replay_buffer.add(
                        clip["observations"].reshape((1, -1)),
                        clip["next_observations"].reshape((1, -1)),
                        clip["actions"].reshape((1, -1)),
                        smoothed_vlm_returns[sorted_ind],
                        [clip["done"]],
                        [clip_info])
                    if hasattr(self.model, 'reward_model2'):
                        self.model.reward_model2.replay_buffer.add(
                            clip["observations"].reshape((1, -1)),
                            clip["next_observations"].reshape((1, -1)),
                            clip["actions"].reshape((1, -1)),
                            smoothed_vlm_returns[sorted_ind],
                            [clip["done"]],
                            [clip_info])
        else:
            for vlm_return, clip, embedding in zip(vlm_returns, clips, clip_embeddings):
                clip_info = clip["info"]
                clip_info["task_rewards"] = clip["rewards"]
                clip_info["clip_embedding"] = embedding.reshape((1, -1))
                self.model.reward_model.replay_buffer.add(
                    clip["observations"].reshape((1, -1)),
                    clip["next_observations"].reshape((1, -1)),
                    clip["actions"].reshape((1, -1)),
                    vlm_return,
                    [clip["done"]],
                    [clip_info])
                if hasattr(self.model, 'reward_model2'):
                    self.model.reward_model2.replay_buffer.add(
                        clip["observations"].reshape((1, -1)),
                        clip["next_observations"].reshape((1, -1)),
                        clip["actions"].reshape((1, -1)),
                        vlm_return,
                        [clip["done"]],
                        [clip_info])
        infos = self.locals["infos"]
        self.model.train_stats_buffer["vlm_reward"].append(
            np.mean(vlm_returns))
        self.model.train_stats_buffer["n_video_clips"].append(
            self.model.reward_model.replay_buffer.size())
        self.model.train_stats_buffer["n_video_clips_added"].append(len(clips))
        return True

# --- RelabelBufferCallback (with pre-relabel synchronization) ---
class RelabelBufferCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Check that reward_model exposes a relabel method
        if hasattr(self.model, 'reward_model') and hasattr(self.model.reward_model, 'relabel'):
            # Invoke the reward model relabel routine
            if self.verbose > 0:
                logger.info("Starting reward model training and relabeling...")
            self.model.reward_model.relabel() # Call the JAX implementation
            if self.verbose > 0:
                logger.info("Reward model training and relabeling finished.")

        return True  # Always return a boolean

class RelabelBufferCallbackShare(BaseCallback):
    def _on_step(self) -> bool:
        # Check that reward_model exposes a relabel method
        if hasattr(self.model, 'reward_model') and hasattr(self.model.reward_model, 'relabel'):

            # <<< Sync the latest agent critic state to the reward model before relabel >>>
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf1_state'):
                if hasattr(self.model.reward_model, 'critic_state_ref'):
                    # Ensure type safety before assignment
                    if isinstance(self.model.policy.qf1_state, TrainState):
                         if self.verbose > 1:
                             logger.info("Syncing agent qf1_state to reward_model.critic_state_ref before relabel")
                         self.model.reward_model.critic_state_ref = self.model.policy.qf1_state
                    else:
                         logger.warning(f"Agent qf1_state type ({type(self.model.policy.qf1_state)}) is not TrainState; cannot sync to reward model.")
                else:
                     logger.warning("Reward model has no 'critic_state_ref' attribute; cannot synchronize.")
            else:
                 logger.warning("Agent policy or qf1_state missing; cannot synchronize.")

            # <<< Sync the latest agent critic state to the reward model before relabel >>>
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf2_state'):
                if hasattr(self.model.reward_model, 'critic_state_ref'):
                    # Ensure type safety before assignment
                    if isinstance(self.model.policy.qf2_state, TrainState):
                         if self.verbose > 1:
                             logger.info("Syncing agent qf2_state to reward_model.critic_state_ref before relabel")
                         self.model.reward_model.critic_state_ref = self.model.policy.qf2_state
                    else:
                         logger.warning(f"Agent qf2_state type ({type(self.model.policy.qf2_state)}) is not TrainState; cannot sync to reward model.")
                else:
                     logger.warning("Reward model has no 'critic_state_ref' attribute; cannot synchronize.")
            else:
                 logger.warning("Agent policy or qf2_state missing; cannot synchronize.")

            # 1. Run reward-model relabel
            if self.verbose > 0:
                logger.info("Starting reward model training and relabeling...")
            try:
                self.model.reward_model.relabel() # Call the JAX implementation
                if self.verbose > 0:
                    logger.info("Reward model training and relabeling finished.")
            except Exception as e:
                logger.error(f"Error calling reward_model.relabel(): {e}")
                import traceback
                traceback.print_exc()
                return True  # Continue training even if relabel fails

            # 2. <<< Update the agent critic state after relabel >>>
            if not hasattr(self.model.reward_model, 'critic_state_ref'):
                 logger.warning("Reward model has no 'critic_state_ref' attribute; cannot update agent critic state.")
                 return True

            updated_critic_state = self.model.reward_model.critic_state_ref

            if not isinstance(updated_critic_state, TrainState):
                 logger.warning(f"Received invalid 'critic_state_ref' type from reward model ({type(updated_critic_state)}); cannot update agent critic state.")
                 # Decide whether to return True or False based on policy
                 # return True  # default: continue training even if sync fails
                 # Alternatively, for stricter handling:
                 # return False  # stop training when synchronization fails
                 # --- choose one behavior above ---
                 return True  # Keep current behavior: log warning and continue


            # Update agent qf1_state
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf1_state'):
                if isinstance(self.model.policy.qf1_state, TrainState):
                    # --- Simplified update logic ---
                    # Skip the heavy opt_state comparison and update when the type matches
                    if self.verbose > 1:
                         logger.info("Updating agent qf1_state with reward-model-trained parameters.")
                    self.model.policy.qf1_state = updated_critic_state
                    # --- end simplified update logic ---

                else:
                    logger.warning(f"Agent 'qf1_state' has invalid type ({type(self.model.policy.qf1_state)}); unable to update.")
            else:
                logger.warning("Agent policy or qf1_state missing; critic state cannot be updated.")

            # Update agent qf2_state
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf2_state'):
                if isinstance(self.model.policy.qf2_state, TrainState):
                    # --- Simplified update logic ---
                    # Skip the heavy opt_state comparison and update when the type matches
                    if self.verbose > 1:
                         logger.info("Updating agent qf2_state with reward-model-trained parameters.")
                    self.model.policy.qf2_state = updated_critic_state
                    # --- end simplified update logic ---

                else:
                    logger.warning(f"Agent 'qf2_state' has invalid type ({type(self.model.policy.qf2_state)}); unable to update.")
            else:
                logger.warning("Agent policy or qf2_state missing; critic state cannot be updated.")

        else:
            # Fallback when reward_model or its relabel method is missing
            logger.warning("Reward model or its relabel method not found on the agent.")

        return True  # Always return a boolean

class RelabelBufferCallbackSeparate(BaseCallback):
    def _on_step(self) -> bool:
        # Check that reward_model exposes a relabel method
        if hasattr(self.model, 'reward_model') and hasattr(self.model.reward_model, 'relabel'):

            # <<< Sync the latest agent critic state to the reward model before relabel >>>
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf1_state'):
                if hasattr(self.model.reward_model, 'critic_state_ref'):
                    # Ensure type safety before assignment
                    if isinstance(self.model.policy.qf1_state, TrainState):
                         if self.verbose > 1:
                             logger.info("Syncing agent qf1_state to reward_model.critic_state_ref before relabel")
                         self.model.reward_model.critic_state_ref = self.model.policy.qf1_state
                    else:
                         logger.warning(f"Agent qf1_state type ({type(self.model.policy.qf1_state)}) is not TrainState; cannot sync to reward model.")
                else:
                     logger.warning("Reward model has no 'critic_state_ref' attribute; cannot synchronize.")
            else:
                 logger.warning("Agent policy or qf1_state missing; cannot synchronize.")

            # <<< Sync the latest agent critic state to the reward model before relabel >>>
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf2_state'):
                if hasattr(self.model.reward_model2, 'critic_state_ref'):
                    # Ensure type safety before assignment
                    if isinstance(self.model.policy.qf2_state, TrainState):
                         if self.verbose > 1:
                             logger.info("Syncing agent qf2_state to reward_model.critic_state_ref before relabel")
                         self.model.reward_model2.critic_state_ref = self.model.policy.qf2_state
                    else:
                         logger.warning(f"Agent qf2_state type ({type(self.model.policy.qf2_state)}) is not TrainState; cannot sync to reward model.")
                else:
                     logger.warning("Reward model has no 'critic_state_ref' attribute; cannot synchronize.")
            else:
                 logger.warning("Agent policy or qf2_state missing; cannot synchronize.")

            # 1. Run reward-model relabel
            if self.verbose > 0:
                logger.info("Starting reward model training and relabeling...")
            try:
                print("relabel1")
                self.model.reward_model.relabel() # Call the JAX implementation
                print("relabel2")
                self.model.reward_model2.relabel() # Call the JAX implementation
                if self.verbose > 0:
                    logger.info("Reward model training and relabeling finished.")
            except Exception as e:
                logger.error(f"Error calling reward_model.relabel(): {e}")
                import traceback
                traceback.print_exc()
                return True  # Continue training even if relabel fails

            # 2. <<< Update the agent critic state after relabel >>>
            if not hasattr(self.model.reward_model, 'critic_state_ref'):
                 logger.warning("Reward model has no 'critic_state_ref' attribute; cannot update agent critic state.")
                 return True
            if not hasattr(self.model.reward_model2, 'critic_state_ref'):
                 logger.warning("Reward model has no 'critic_state_ref' attribute; cannot update agent critic state.")
                 return True

            updated_critic_state = self.model.reward_model.critic_state_ref
            updated_critic_state2 = self.model.reward_model2.critic_state_ref

            if not isinstance(updated_critic_state, TrainState):
                 logger.warning(f"Received invalid 'critic_state_ref' type from reward model ({type(updated_critic_state)}); cannot update agent critic state.")
                 # Decide whether to return True or False based on policy
                 # return True  # default: continue training even if sync fails
                 # Alternatively, for stricter handling:
                 # return False  # stop training when synchronization fails
                 # --- choose one behavior above ---
                 return True  # Keep current behavior: log warning and continue
            if not isinstance(updated_critic_state2, TrainState):
                 logger.warning(f"Received invalid 'critic_state_ref' type from reward model ({type(updated_critic_state2)}); cannot update agent critic state.")
                 # Decide whether to return True or False based on policy
                 # return True  # default: continue training even if sync fails
                 # Alternatively, for stricter handling:
                 # return False  # stop training when synchronization fails
                 # --- choose one behavior above ---
                 return True


            # Update agent qf1_state
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf1_state'):
                if isinstance(self.model.policy.qf1_state, TrainState):
                    # --- Simplified update logic ---
                    # Skip the heavy opt_state comparison and update when the type matches
                    if self.verbose > 1:
                         logger.info("Updating agent qf1_state with reward-model-trained parameters.")
                    self.model.policy.qf1_state = updated_critic_state
                    # --- end simplified update logic ---

                else:
                    logger.warning(f"Agent 'qf1_state' has invalid type ({type(self.model.policy.qf1_state)}); unable to update.")
            else:
                logger.warning("Agent policy or qf1_state missing; critic state cannot be updated.")

            # Update agent qf2_state
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'qf2_state'):
                if isinstance(self.model.policy.qf2_state, TrainState):
                    # --- Simplified update logic ---
                    # Skip the heavy opt_state comparison and update when the type matches
                    if self.verbose > 1:
                         logger.info("Updating agent qf2_state with reward-model-trained parameters.")
                    self.model.policy.qf2_state = updated_critic_state2
                    # --- end simplified update logic ---

                else:
                    logger.warning(f"Agent 'qf2_state' has invalid type ({type(self.model.policy.qf2_state)}); unable to update.")
            else:
                logger.warning("Agent policy or qf2_state missing; critic state cannot be updated.")

        else:
            # Fallback when reward_model or its relabel method is missing
            logger.warning("Reward model or its relabel method not found on the agent.")

        return True  # Always return a boolean

class EvalCallback(SB3EvalCallback):
    """Store and upload render arrays in addition to policy evaluation."""
    def _init_callback(self, n_video_per_call=1) -> None:
        super()._init_callback()
        run_dir = os.path.dirname(self.log_path)
        self.gif_path = os.path.join(run_dir, 'evaluation_trajectories')
        os.makedirs(self.gif_path, exist_ok=True)
        self.n_video_per_call = n_video_per_call

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """In addition to logging success, save render arrays"""
        super()._log_success_callback(locals_=locals_, globals_=globals_)
        env_id = locals_["i"]
        info = locals_["info"]
        ep_count = sum(locals_["episode_counts"])
        buffer = self.render_array_buffer[env_id]
        if len(self.render_array_buffer[env_id]) == 0:
            # The eval_env has been reset
            # Need to retrieve the first render array
            initial_array = self.eval_env.reset_infos[env_id]["render_array"]
            buffer.append(initial_array)
        if locals_["done"]:
            buffer.append(info["terminal_render_array"])
            if ep_count < self.n_video_per_call:
                imageio.mimsave(
                    self.gif_path + f'/{self.model.num_timesteps}_{ep_count}.gif',
                    buffer,
                    duration=0.04,
                    loop=0)
            self.render_array_buffer[env_id] = [info["render_array"]]
            if ep_count == 0:
                wandb_video = wandb.Video(
                self.gif_path + f'/{self.model.num_timesteps}_{ep_count}.gif',
                fps=24,
                format="gif",)
                wandb.log({str(self.model.num_timesteps): wandb_video},
                            step=self.model.num_timesteps)
        else:
            buffer.append(info["render_array"])

    def _on_step(self) -> bool:
        self.render_array_buffer = defaultdict(list)
        # return super()._on_step()
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            # Evaluate the policy
            episode_rewards, episode_lengths, episode_successes = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                assert isinstance(episode_successes, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_success = np.mean(episode_successes)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_success", float(mean_success))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def _on_training_start(self) -> None:
        self._on_step()
        return super()._on_training_start()
