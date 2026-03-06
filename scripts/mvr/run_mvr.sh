#!/bin/bash
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Capture command-line task names and seeds
SEEDS=("${@:1}")  # Seeds collected from command-line arguments
source .venv/bin/activate
# Environment variables for JAX/CPU/GPU
# For a single task, consider unsetting OMP_NUM_THREADS for higher throughput
# 8 threads, TD3, ~1.8k fps; no limit, TD3, >2k fps
export OMP_NUM_THREADS=8
export JAX_PLATFORM_NAME=gpu
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
TASKS="[h1hand_stand,h1hand_slide]"
DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Using unified timestamp: $DATE_TIME"
HYDRA_FULL_ERROR=1 python src/training/train.py\
    --config-name=algo/vlm_tqc \
    --multirun \
    seed=0,1,2 \
    run_name_prefix="mvr-h1"\
    date_time=$DATE_TIME \
    collect_clip_callback.vlm._target_=vlms.ViCLIP \
    collect_clip_callback.vlm.pretrained=ckpts/ViCLIP/ViCLIP-L_InternVid-FLT-10M.pth\
    collect_clip_callback.encoding_batch_size=1\
    agent.reward_model_class=reward_models.per_step_ranking_base_model2.PerStepRankingBased2\
    agent.policy=MlpPolicy \
    agent._target_=src.algorithms.vlm.vlm_tqc.VLMTQC\
    agent.vlm_reward_scale=0.1\
    env_config.camera_azimuths=[90,180,270,0]\
    resume=true \
    wandb_mode=offline \
    wandb_project_name=mvr-rl\
    tasks=$TASKS\
    total_timesteps=10000000
