import gymnasium
from .customized_humanoid import CustomizedHumanoidEnv
from .metaworld_wrapper import MetaWorldWrapper
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.envs.registration import register

register(
    id="CustomizedHumanoid-v4",
    entry_point=CustomizedHumanoidEnv,
    max_episode_steps=1000,
)
from humanoid_bench.env import ROBOTS, TASKS
from . import customized_humanoid_bench
for robot in ROBOTS:
    if robot == "g1" or robot == "digit":
        control = "torque"
    else:
        control = "pos"
    for task, task_info in TASKS.items():
        task_info = task_info()
        kwargs = task_info.kwargs.copy()
        kwargs["robot"] = robot
        kwargs["control"] = control
        kwargs["task"] = task
        register(
            id=f"{robot}-{task}-customized-v0",
            entry_point=customized_humanoid_bench.HumanoidEnv,
            max_episode_steps=task_info.max_episode_steps,
            kwargs=kwargs,
        )

# Metaworld
for task_name in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE.keys():
    env_id = f"{task_name}"
    register(
        id=env_id,
        entry_point=MetaWorldWrapper,  
        kwargs={"task_name": task_name},
        max_episode_steps=500,
    )