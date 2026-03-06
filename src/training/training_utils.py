import gc
import glob
import os
import random
import time
from typing import Dict, List, Optional, Tuple
import zipfile

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
import torch

def create_wrapper_from_config(env_config, training=True):
    """Create environment wrapper callables based on configuration."""
    def wrapper_fn(env):
        if training is False:
            env = RenderingWrapper(env)
        
        # Apply configured wrappers if specified
        if env_config.get('wrapper_class'):
            wrapper_class_path = env_config['wrapper_class']
            wrapper_kwargs = env_config.get('wrapper_kwargs', {})
            
            # Dynamically import the wrapper class
            if wrapper_class_path == 'env.r3m_state_wrapper.R3MStateWrapper':
                from env.r3m_state_wrapper import R3MStateWrapper
                print(f"Applying R3MStateWrapper with kwargs: {wrapper_kwargs}")
                env = R3MStateWrapper(env, **wrapper_kwargs)
            
        return env
    return wrapper_fn


class RenderingWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Render observations automatically after calling step() and reset().

    The rendered RGB array is stored in the info dict under key render_array.
    """

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["render_array"] = np.array(self.render())
        return obs, reward, terminated, truncated, info
    
    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None) -> Tuple[NDArray, Dict]:
        obs, info = super().reset(seed=seed, options=options)
        info["render_array"] = np.array(self.render())
        return obs, info


def set_seed(seed):
    random.seed(seed)  # Set the random seed for Python's random module
    np.random.seed(seed)  # Set the random seed for numpy
    torch.manual_seed(seed)  # Set the random seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set the random seed for the GPU if available
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs if using multiple
    torch.backends.cudnn.deterministic = True  # Make CuDNN backend deterministic
    torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuning for deterministic results
    torch.use_deterministic_algorithms(True)  # Ensure deterministic algorithms are used in PyTorch

    # Disable certain operations that might lead to non-deterministic behavior
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Ensure that third-party libraries also set their seeds
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)  # Set the random seed for TensorFlow
    except ImportError:
        pass
    
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)  # Set the random seed for TPU (used with PyTorch XLA)
    except ImportError:
        pass

    # Ensure other libraries (e.g., scikit-learn) also use the same random seed
    try:
        import sklearn
        sklearn.utils.validation._NUMPY_RANDOM_SEED = seed
    except ImportError:
        pass

def set_egl_env_vars() -> None:
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"

def set_osmesa_env_vars() -> None:
    os.environ["MUJOCO_GL"] = "osmesa"


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        return jnp.where(
            (1 - progress_remaining) > end_fraction,
            end,
            start + (1 - progress_remaining) * (end - start) / end_fraction
        )
    return func


def parse_linear_scheduler(value_in_config):
    """A helper function that parse lienar schedulers from config files.
    
    Arguments:
        value_in_config: a float/int or str. If it is a float/int, use this value directly; otherwise, it should be in the form of 'lin_start_end', where start and end are the start value and end value of linear scheduling.
    """
    if isinstance(value_in_config, str):
        assert value_in_config.startswith("lin")
        lr_configs = value_in_config.split("_")
        start_value = float(lr_configs[1])
        if len(lr_configs) == 2:
            end_value = 0
            frac = 1
        else:
            end_value = float(lr_configs[2])
            frac = 1
            if len(lr_configs) == 4:
                frac = float(lr_configs[3])
        return get_linear_fn(start_value, end_value, frac)
    else:
        return value_in_config

def verify_checkpoint_integrity(checkpoint_path: str) -> bool:
    """Validate the integrity of a checkpoint archive."""
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zip_file:
            bad_file = zip_file.testzip()
            if bad_file:
                print(f"    ZIP archive corrupted: {bad_file}")
                return False

            required_files = ['data', 'pytorch_variables.pth', '_stable_baselines3_version']
            file_list = zip_file.namelist()

            for required_file in required_files:
                if required_file not in file_list:
                    print(f"    Missing required file: {required_file}")
                    return False

            data_info = zip_file.getinfo('data')
            if data_info.file_size == 0:
                print("    data entry is empty")
                return False

            return True

    except zipfile.BadZipFile:
        print("    Not a valid ZIP archive")
        return False
    except Exception as e:
        print(f"    Error during verification: {e}")
        return False


def wait_for_checkpoint_completion(checkpoint_path: str, max_wait_time: int = 300, check_interval: int = 5) -> bool:
    """Wait for a checkpoint file to finish writing and validate it."""
    print(f"Waiting for checkpoint to finish writing: {checkpoint_path}")
    
    start_time = time.time()
    last_size = 0
    stable_count = 0

    while time.time() - start_time < max_wait_time:
        if not os.path.exists(checkpoint_path):
            print(f"  File not found yet, waiting... ({time.time() - start_time:.1f}s)")
            time.sleep(check_interval)
            continue

        current_size = os.path.getsize(checkpoint_path)
        if current_size == last_size and current_size > 0:
            stable_count += 1
            print(f"  File size stable ({current_size / 1024 / 1024:.1f}MB), stability count: {stable_count}/3")

            if stable_count >= 3:
                if verify_checkpoint_integrity(checkpoint_path):
                    print(f"  ✅ Checkpoint verified: {checkpoint_path}")
                    return True
                else:
                    print("  ❌ Checkpoint appears corrupted, continuing to wait...")
                    stable_count = 0
        else:
            if current_size != last_size:
                print(f"  File size change: {last_size / 1024 / 1024:.1f}MB -> {current_size / 1024 / 1024:.1f}MB")
            stable_count = 0
            last_size = current_size

        time.sleep(check_interval)

    print(f"  ⚠️ Timed out after {max_wait_time}s; file may still be incomplete")
    return False


def find_completed_tasks(base_output_dir: str, run_name_suffix: str, tasks: List[str], 
                        task_names: dict, seed: int, algo: str, date_time: str) -> Tuple[List[int], Optional[str]]:
    """Locate completed tasks and their checkpoint files."""
    completed_tasks = []
    last_checkpoint_path = None

    print(f"\n=== Scanning completed tasks (seed {seed}) ===")
    
    for i, task in enumerate(tasks):
        task_name = task_names[task]
        run_name = f"benchmark-{algo}-{task}-seed{seed}-id{i}{run_name_suffix}"
        
        # Construct task directory path
        task_dir = os.path.join(base_output_dir, run_name_suffix.strip('-'), date_time, run_name, task_name)
        checkpoint_path = os.path.join(task_dir, "checkpoints.zip")

        print(f"Checking task {i+1}/{len(tasks)}: {task_name} (seed {seed})")
        print(f"  Checkpoint path: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            if verify_checkpoint_integrity(checkpoint_path):
                print(f"  ✅ Task {i+1} ({task_name}) completed with valid checkpoint")
                completed_tasks.append(i)
                last_checkpoint_path = checkpoint_path
            else:
                print(f"  ❌ Task {i+1} ({task_name}) checkpoint corrupted")
                break
        else:
            print(f"  ❌ Task {i+1} ({task_name}) incomplete (missing checkpoint)")
            break

    return completed_tasks, last_checkpoint_path


def determine_resume_point(config: DictConfig, base_output_dir: str) -> Tuple[int, str, Optional[str]]:
    """Determine where to resume training."""
    if not config.get('resume', False):
        print("Resume disabled; starting from scratch")
        return 0, "", None

    print(f"\n🔄 Resume enabled (seed {config.seed})")
    
    # Determine run directory
    if config.get('resume_run_dir'):
        run_dir = config.resume_run_dir
        if not os.path.exists(run_dir):
            print(f"❌ Specified run directory does not exist: {run_dir}")
            return 0, "", None
        date_time = os.path.basename(run_dir)
    else:
        # Automatically locate the most recent run directory
        pattern = os.path.join(base_output_dir, config.run_name_prefix.strip('-'), "*")
        run_dirs = glob.glob(pattern)
        if not run_dirs:
            print("❌ No matching run directory found")
            return 0, "", None
        
        run_dirs.sort(key=os.path.getmtime, reverse=True)
        run_dir = run_dirs[0]
        date_time = os.path.basename(run_dir)

    # Identify completed tasks
    tasks = config.tasks
    task_names = config.task_names
    
    completed_tasks, last_checkpoint_path = find_completed_tasks(
        base_output_dir, config.run_name_suffix, tasks, 
        task_names, config.seed, config.algo, date_time
    )

    if not completed_tasks:
        print("❌ No completed tasks detected; starting from scratch")
        return 0, "", date_time

    # Determine starting task
    if config.get('resume_from_task') is not None:
        start_task_index = config.resume_from_task
        if start_task_index < 0 or start_task_index >= len(config.tasks):
            print(f"❌ Invalid resume task index: {start_task_index}")
            return 0, "", date_time
        
        if start_task_index == 0:
            prev_agent_path = ""
        else:
            prev_agent_path = ""
    else:
        # Resume from the next incomplete task
        start_task_index = len(completed_tasks)
        if start_task_index >= len(tasks):
            print(f"🎉 All tasks for seed {config.seed} are complete!")
            return start_task_index, "", date_time
        
        prev_agent_path = ""

    print(f"Resuming from task {start_task_index+1}")
    return start_task_index, prev_agent_path, date_time

def clear_gpu_memory():
    """Clear GPU memory across PyTorch and JAX stacks."""
    # 1. Trigger Python garbage collection
    gc.collect()
    
    # 2. Release PyTorch memory
    torch.cuda.synchronize()  # synchronize first
    torch.cuda.empty_cache()  # then clear cache
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # 3. Release JAX memory
    jax.devices()  # ensure devices are initialized
    # Synchronize pending GPU work
    jax.device_get(jax.numpy.array(0))
    # Clear caches
    jax.clear_caches()
    try:
        jax.clear_backends()
    except AttributeError:
        pass
