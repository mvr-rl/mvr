from datetime import datetime
import multiprocessing as mp
import os
import os.path as osp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import flax
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import optax
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import wandb

import env
from src.training.callbacks import EvalCallback, LogTrainingStats, WandbCallback
from src.training.training_utils import (
    RenderingWrapper,
    create_wrapper_from_config,
    determine_resume_point,
    parse_linear_scheduler,
    set_egl_env_vars,
    set_osmesa_env_vars,
    set_seed,
    wait_for_checkpoint_completion,
)
from src.utils.subproc_vec_env import SubprocVecEnv, VecNormalize

mp.set_start_method('spawn', force=True)




# Register OmegaConf resolvers
OmegaConf.register_new_resolver("linear_scheduling", lambda v: parse_linear_scheduler(v))
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("plus_one", lambda x: x + 1)
OmegaConf.register_new_resolver("nn", lambda v: getattr(flax.linen, v))
OmegaConf.register_new_resolver("optax", lambda v: getattr(optax, v))
def run_single_task(config: OmegaConf, task: str, task_index: int, prev_agent_path: str = "") -> str:
    """
    Run training for a single task.
    """
    print(f"Starting task: {task}")
    
    # Configure rendering backend
    if config.rendering_backend == "egl":
        set_egl_env_vars()
    elif config.rendering_backend == "osmesa":
        set_osmesa_env_vars()
    else:
        raise NotImplementedError

    # Build environments
    task_name = config.task_names[task]
    
    # Prepare environment kwargs
    env_kwargs = OmegaConf.to_container(config.env_config, resolve=True) if config.get('env_config') else {}
    vec_env_kwargs = {}
    if config.get('video_sampling_config'):
        vec_env_kwargs["video_sampling_configs"] = OmegaConf.to_container(config.video_sampling_config, resolve=True)

    # Instantiate wrappers defined in the config
    wrapper_fn_train = create_wrapper_from_config(config.env_config, training=True)
    wrapper_fn_eval = create_wrapper_from_config(config.env_config, training=False)
    
    env = make_vec_env(
        task_name,
        n_envs=config.num_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=wrapper_fn_train,
        env_kwargs=env_kwargs,
        vec_env_kwargs=vec_env_kwargs
    )

    eval_env = make_vec_env(
        task_name,
        n_envs=2,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=wrapper_fn_eval,
        env_kwargs=env_kwargs
    )

    # Apply VecNormalize if specified in config (e.g., for Simba)
    if config.get('normalize'):
        normalize_config = config.normalize
        if normalize_config.get('norm_obs', False):
            print(f"Applying observation normalization: {normalize_config}")
            env = VecNormalize(
                env, 
                norm_obs=normalize_config.get('norm_obs', False),
                norm_reward=normalize_config.get('norm_reward', False),
                training=True
            )
            eval_env = VecNormalize(
                eval_env,
                norm_obs=normalize_config.get('norm_obs', False), 
                norm_reward=normalize_config.get('norm_reward', False),
                training=False
            )

    set_seed(config.seed)
    
    # Initialize WandB
    algo = config.algo
    seed = config.seed
    run_name = f"{config.run_name_prefix}#{algo}#{task}#seed{seed}#id{task_index}"

    run = wandb.init(
        project=config.wandb_project_name,
        name=run_name,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        mode=config.wandb_mode,
        tags=[task_name]
    )

    # Prepare output directory
    if config.get('date_time'):
        date_time = config.date_time
    else:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Directory layout: outputs/<algo>/<timestamp>/<run_name>/<env_name>
    output_dir = os.path.join(
        config.get('outputs_dir', 'outputs'),
        algo,
        date_time,  # timestamp acts as parent folder
        run_name,   # run name as intermediate folder
        task_name
    )
    os.makedirs(output_dir, exist_ok=True)

    # Configure callbacks
    eval_freq = config.eval_freq
    num_envs = config.num_envs
    eval_render = config.eval_render
    
    callback_list = [
        EvalCallback(
            eval_env=eval_env,
            log_path=output_dir,
            eval_freq=eval_freq // num_envs,
            render=eval_render,
            n_eval_episodes=2 if eval_render else 10,
        ),
        LogTrainingStats(),
        WandbCallback(
            model_save_path=None,
            model_save_freq=0,
            verbose=2,
        )
    ]

    # Register VLM-specific callbacks if needed
    if config.get('use_vlm', False):
        print("Enabling VLM-specific callbacks")
        config.agent.info_keys_to_print += config.vlm_related_stats
        callback_list.extend([
            EveryNTimesteps(
                config.get('video_collect_freq', 10000),
                instantiate(config.collect_clip_callback)(env=env)
            ),
            EveryNTimesteps(
                config.get('relabel_freq', 50000),
                instantiate(config.relabel_buffer_callback)
            )
        ])
    # If use_vlm is False, callbacks remain unchanged

    # Instantiate or load the agent
    agent_config = config.agent
    agent = instantiate(agent_config, _convert_="all")(env=env, prev_agent=None)
        
    # Configure logger
    new_logger = configure(osp.join(output_dir, "tb_logs"), ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # Run training
    if not eval_render:
        learn_config = config.learn
        log_interval = learn_config.get('log_interval', 10) if isinstance(learn_config, dict) else getattr(learn_config, 'log_interval', 10)
        
        agent.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callback_list),
            log_interval=log_interval,
            progress_bar=True
        )
        
        # Persist checkpoint
        checkpoint_path = osp.join(output_dir, "checkpoints.zip")
        agent.save(checkpoint_path)
        
    # Clean up references
    agent.prev_agent = None
    if hasattr(agent, 'policy') and hasattr(agent.policy, 'prev_policy'):
        agent.policy.prev_policy = None
    env.close()
    eval_env.close()
    run.finish()
    return osp.join(output_dir, "checkpoints.zip") if not config.get('eval_render', False) else ""


def run_task_sequence_training(config: OmegaConf):
    
    # Use a unified timestamp across all tasks in this multi-task run
    if config.get('date_time') is None:
        config.date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"📅 Unified multi-task timestamp: {config.date_time}")
    
    base_output_dir = config.get('outputs_dir', 'outputs')
    start_task_index, _, resume_date_time = determine_resume_point(config, base_output_dir)
    prev_agent_path = ""  # always start fresh; do not load prior checkpoint
    
    # Retrieve task list
    tasks = config.tasks
    
    # Check whether all tasks are already complete
    if start_task_index >= len(tasks):
        print(f"All tasks for seed {config.seed} are complete; skipping execution")
        return

    # Use the recovered timestamp when resuming
    if config.get('resume', False) and resume_date_time:
        config.date_time = resume_date_time
        print(f"📅 Resuming training with timestamp: {config.date_time}")

    # Iterate over remaining tasks
    tasks_to_run = tasks[start_task_index:]
    for relative_i, task in enumerate(tasks_to_run):
        i = start_task_index + relative_i
        print(f"\n=== Task {i+1}/{len(tasks)}: {task} ===")
        
        # Train the current task
        checkpoint_path = run_single_task(config, task, i, prev_agent_path)
        
        # Report progress
        completed_count = i + 1
        print(f"📊 Progress: {completed_count}/{len(tasks)} tasks completed")


@hydra.main(config_path="../../configs", config_name="algo/tqc", version_base="1.1")
def main(config: OmegaConf):
    print("=== Config ===")
    print(OmegaConf.to_yaml(config))
    
    # Determine execution mode from the config
    tasks = config.get('tasks', [])
    print(f"📋 Detected {len(tasks)} tasks; launching sequential training")
    run_task_sequence_training(config)


if __name__ == '__main__':
    main() 
