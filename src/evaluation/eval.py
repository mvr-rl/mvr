import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from stable_baselines3.common.callbacks import CallbackList,EveryNTimesteps
import env
from src.utils.subproc_vec_env import SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from src.training.callbacks import LogTrainingStats,WandbCallback, RelabelBufferCallback,EvalCallback
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.logger import configure
import os.path as osp
from omegaconf import OmegaConf
from hydra.utils import instantiate
import hydra
import reward_models
from src.training.training_utils import set_egl_env_vars, set_osmesa_env_vars, set_seed, RenderingWrapper, parse_linear_scheduler

OmegaConf.register_new_resolver(
    "linear_scheduling", lambda v: parse_linear_scheduler(v))
OmegaConf.register_new_resolver(
    "reward_model", lambda v: getattr(reward_models, v))
@hydra.main(
    config_path="../../configs", config_name="tqc", version_base="1.1")
def main(config: OmegaConf):
    print(OmegaConf.to_yaml(config))
    if config.rendering_backend == "egl":
        set_egl_env_vars()
    elif config.rendering_backend == "osmesa":
        set_osmesa_env_vars()
    else:
        raise NotImplementedError

    env = make_vec_env(
        config.task.name,
        n_envs=config.num_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=OmegaConf.to_container(config.env_config, resolve=True),
        vec_env_kwargs={"video_sampling_configs":OmegaConf.to_container(config.video_sampling_config, resolve=True)})
    #env = VecNormalize(env, norm_reward=True)

    eval_env = make_vec_env(
        config.task.name,
        n_envs=2,#10,
        seed=config.seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=RenderingWrapper,  # Pass the custom wrapper chain function
        env_kwargs=OmegaConf.to_container(config.env_config, resolve=True))
    #eval_env = VecNormalize(eval_env, training=False)

    set_seed(config.seed)
    run = wandb.init(
        project=config.wandb_project_name+"-eval",
        name=config.run_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        mode=config.wandb_mode,
        tags=[config.task.name])
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    callback_list = [
        EvalCallback(
            eval_env=eval_env,
            log_path=output_dir,
            eval_freq=config.eval_freq // config.num_envs,
            n_eval_episodes=100),
        LogTrainingStats(),
        WandbCallback(
            model_save_path=None,#osp.join(output_dir, "checkpoints"),
            model_save_freq=0,#config.eval_freq,
            verbose=2,)]
    if config.agent._target_=="vlm_tqc.VLMTQC" or config.agent._target_=="wa_vlm_tqc.WATQC":
        config.agent.info_keys_to_print += config.vlm_related_stats
        callback_list.extend([
            EveryNTimesteps(
                config.video_collect_freq,
                instantiate(config.collect_clip_callback)(env=env)),
            EveryNTimesteps(config.relabel_freq, RelabelBufferCallback())])
    agent = instantiate(config.agent, _convert_="all")(env=env)
    # if load agent
    agent = agent.load(config.ckpt_path,env=env,custom_objects={"reward_model_class": reward_models.PerStepFittingBased2,"learning_rate": parse_linear_scheduler(config.agent.learning_rate)})
    new_logger = configure(
        osp.join(output_dir, "tb_logs"), ["stdout", "tensorboard"])
    agent.set_logger(new_logger)
    agent.learn(total_timesteps=1,
                callback=CallbackList(callback_list),
                log_interval=config.learn.log_interval)
    print("Training finished")
    env.close()
    # run.finish()
if __name__ == '__main__':
#   app.run(main)
    main(None)