from sbx import TQC as sbxTQC
from sbx.tqc.tqc import ConstantEntropyCoef, EntropyCoef
from stable_baselines3.common.utils import safe_mean

from sbx.common.type_aliases import RLTrainState
import optax
from optax import schedules, adam
from flax.training.train_state import TrainState
import jax
from gymnasium import spaces
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple, Type, Union
import numpy as np

import jax.numpy as jnp
from sbx.tqc.policies import TQCPolicy as sbxTQCPolicy
from sbx.tqc.policies import SimbaTQCPolicy as sbxSimbaTQCPolicy
from sbx.tqc.policies import ContinuousCritic as Critic, SquashedGaussianActor as Actor
from stable_baselines3.common.type_aliases import Schedule, GymEnv, MaybeCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.torch_layers import NatureCNN
from functools import partial

import flax
import flax.linen as nn
from jax.typing import ArrayLike

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp
from sbx.tqc.policies import TQCPolicy

class TQCPolicy(sbxTQCPolicy):
    
    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
        key, dropout_key1, dropout_key2, self.key = jax.random.split(key, 4)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])

        action = jnp.array([self.action_space.sample()])

        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Use inject_hyperparams for SBX 0.21+ compatibility
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1), **self.optimizer_kwargs
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optimizer_class,
        )

        self.qf = Critic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            output_dim=self.n_quantiles,
            activation_fn=self.activation_fn,
        )

        # Use inject_hyperparams for Q-function optimizers as well
        qf_optimizer_class = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1), **self.optimizer_kwargs
        )

        self.qf1_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            tx=qf_optimizer_class,
        )
        self.qf2_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            tx=qf_optimizer_class,
        )
        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

        return key

class TQC(sbxTQC):
    policy_aliases: ClassVar[Dict[str, Type[TQCPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": TQCPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": TQCPolicy,
        # CNN policy for pixel observations - reuse standard TQCPolicy plus CNN feature extractor
        "CnnPolicy": TQCPolicy,
    }
    def __init__(self, *args, n_experts=None, info_keys_to_print=[], prev_agent=None, expert_loss_coeff=None, **kwargs):
        self.info_keys_to_print = info_keys_to_print
        super().__init__(*args, **kwargs)
    
    def _setup_lr_schedule(self) -> None:
        """Use the SBX 0.21 learning rate scheduling mechanism."""
        # Remove the custom implementation; rely on the parent logic
        # This automatically leverages stable_baselines3.common.utils.FloatSchedule
        super()._setup_lr_schedule()

    def dump_logs(self) -> None:
        if len(self.ep_info_buffer) > 0 and len(self.train_stats_buffer)>0:
            for key in self.info_keys_to_print:
                if key in self.train_stats_buffer:
                    if len(self.train_stats_buffer[key]) > 0:
                        self.logger.record(
                            f"rollout/{key}_mean",
                            safe_mean(self.train_stats_buffer[key]),)
        # Pass the number of timesteps for tensorboard
        super().dump_logs()

    def train(self, gradient_steps, batch_size):
        self._update_learning_rate()
        super().train(gradient_steps, batch_size)

    def _update_learning_rate(self, optimizers=None, learning_rate=None, name="learning_rate") -> None:
        """
        Use the SBX 0.21 standard learning-rate update mechanism.

        This implementation is now fully compatible with the built-in scheduler.
        """
        # When no explicit arguments, rely on the default SBX update
        if optimizers is None and learning_rate is None:
            # Compute current LR (using FloatSchedule)
            progress = self._current_progress_remaining
            current_lr = self.lr_schedule(progress)

            # Log current learning rate
            self.logger.record("train/learning_rate", current_lr)
        else:
            # Handle the new SBX API call signature
            if optimizers is not None and learning_rate is not None:
                if not isinstance(optimizers, list):
                    optimizers = [optimizers]
                for optimizer in optimizers:
                    # Optimizer needs to be defined via inject_hyperparams
                    if hasattr(optimizer, 'hyperparams'):
                        optimizer.hyperparams["learning_rate"] = learning_rate

                # Record the provided learning rate
                self.logger.record(f"train/{name}", learning_rate)
    
    def _setup_model(self) -> None:
        # Overload the setup function of sbxTQC to remove an assertion
        super(sbxTQC, self)._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )
            #assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
                # Default initial value of ent_coef when learned
                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert ent_coef_init > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            # Use inject_hyperparams for entropy coefficient optimizer
            ent_optimizer_class = optax.inject_hyperparams(optax.adam)(
                learning_rate=self.lr_schedule(1)
            )

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=ent_optimizer_class,
            )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)
