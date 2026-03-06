from typing import Any, Dict, List, Tuple, TypeVar, Union
import numpy as np
from src.algorithms.base.sbx_tqc import TQC
from stable_baselines3.common.buffers import ReplayBuffer
import pathlib
import io
import torch as th
import warnings
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, NamedTuple
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env
from stable_baselines3.common.base_class import check_for_correct_spaces, _convert_space, get_system_info, SelfBaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, TensorDict
from hydra.utils import get_class

class VLMReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_vlm_rewards = np.zeros_like(self.rewards)

    def add(
        self,
        obs: np.typing.NDArray,
        next_obs: np.typing.NDArray,
        action: np.typing.NDArray,
        reward: np.typing.NDArray,
        done: np.typing.NDArray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if 'pred_vlm_reward' in infos[0]:
            pred_vlm_reward = np.array(
                [info['pred_vlm_reward'] for info in infos])
            self.pred_vlm_rewards[self.pos] = pred_vlm_reward
        super().add(obs, next_obs, action, reward, done, infos)

SelfVLMTQC = TypeVar("SelfVLMTQC", bound="VLMTQC")

class VLMTQC(TQC):
    '''VLM rewarded TQC with learned per-step rewards.
    '''
    def __init__(self,
                 *args,
                 reward_model_class,
                 video_length=64,
                 reward_learning_buffer_size=10_000,
                 relabel_freq=100_000,
                 vlm_reward_scale=1,
                 n_top=10,
                 temperature=0.1,
                 alignment_loss_weight=1,
                 replay_buffer_class=VLMReplayBuffer,
                 **kwargs):
        self.video_length = video_length
        self.reward_learning_buffer_size = reward_learning_buffer_size
        self.relabel_freq = relabel_freq
        self.reward_model_class = reward_model_class
        self.vlm_reward_scale = vlm_reward_scale
        self.n_top = n_top
        self.temperature = temperature
        self.alignment_loss_weight = alignment_loss_weight
        kwargs["replay_buffer_class"] = replay_buffer_class
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if isinstance(self.reward_model_class, str):
            ResolvedRewardModelClass = get_class(self.reward_model_class)
        else:
            ResolvedRewardModelClass = self.reward_model_class
        self.reward_model = ResolvedRewardModelClass(self)

    def _excluded_save_params(self) -> List[str]:
        default_list = super()._excluded_save_params()
        default_list.extend(["reward_model"])
        return default_list

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        state_dicts.append("reward_model")
        return state_dicts, saved_pytorch_variables

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        per_step_reward = self.reward_model.predict(
                self._last_obs, buffer_action)
        mean_per_step_reward = per_step_reward.mean()
        self.train_stats_buffer["pred_vlm_reward"].append(
                mean_per_step_reward)
        if self.reward_model.trained:
            total_rewards = per_step_reward + reward
            for ind, info in enumerate(infos):
                info['pred_vlm_reward'] = per_step_reward[ind]
        else:
            total_rewards = reward
        super()._store_transition(replay_buffer, buffer_action, new_obs, total_rewards, dones, infos)

    @classmethod
    def load(  # noqa: C901
        cls: type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            saved_net_arch = data["policy_kwargs"].get("net_arch")
            if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            reward_model_class=data["reward_model_class"],
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load policies saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        except ValueError as e:
            # Patch to load DQN policies saved using SB3 < 2.4.0
            # The target network params are no longer in the optimizer
            # See https://github.com/DLR-RM/stable-baselines3/pull/1963
            saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]  # type: ignore[index]
            n_params_saved = len(saved_optim_params)
            n_params = len(model.policy.optimizer.param_groups[0]["params"])
            if n_params_saved == 2 * n_params:
                # Truncate to include only online network params
                params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[:n_params]  # type: ignore[index]

                model.set_parameters(params, exact_match=True, device=device)
                warnings.warn(
                    "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                    "we truncated the optimizer state so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/pull/1963 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model