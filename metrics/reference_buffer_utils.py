#!/usr/bin/env python3
"""
Utilities for rebuilding the reference buffer after loading a model when it is empty.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_reference_buffer_status(model):
    """Report the status of the reference buffer."""
    reward_model = model.reward_model
    
    print("\n📊 Reference buffer status check:")
    print(f"  reference_rewards present: {reward_model.reference_rewards is not None}")
    print(f"  trained flag: {reward_model.trained}")
    print(f"  n_updates: {reward_model.n_updates}")
    print(f"  n_top: {reward_model.n_top}")
    
    if reward_model.reference_rewards is not None:
        print(f"  reference size: {reward_model.reference_rewards.shape[0]}")
        print(f"  reference_rewards_flattened size: {len(reward_model.reference_rewards_flattened)}")
        return True
    else:
        print("  ⚠️ Reference buffer is empty, rebuilding is required!")
        return False

def rebuild_reference_buffer(model, env, n_episodes=30):
    """
    Rebuild the reference buffer so the reward model regains its benchmarking baseline.
    Fails fast when rebuilding is unsuccessful.

    Args:
        model: VLMTQC model instance.
        env: Environment instance.
        n_episodes: Number of episodes to collect.

    Raises:
        RuntimeError: If the rebuild fails.
    """
    print(f"\n🔄 Rebuilding reference buffer (collecting {n_episodes} episodes)...")
    
    reward_model = model.reward_model
    obs = env.reset()
    
    # Collect trajectories into the replay buffer
    collected_steps = 0
    episodes_completed = 0
    
    pbar = tqdm(total=n_episodes, desc="🎯 Rebuilding reference", unit="episodes")
    
    while episodes_completed < n_episodes:
        # Use deterministic policy to reduce randomness
        action, _ = model.predict(obs, deterministic=True)
        
        # Interact with the environment
        new_obs, rewards, dones, infos = env.step(action)
        
        # Store transitions into the reward model replay buffer.
        # The reward model expects sequences of length ``video_length``.

        # Observation sequence: repeat the current observation ``video_length`` times
        obs_seq = np.tile(obs, reward_model.video_length).reshape(1, -1)
        next_obs_seq = np.tile(new_obs, reward_model.video_length).reshape(1, -1)
        
        # Action sequence: repeat the action ``video_length`` times
        action_seq = np.tile(action, reward_model.video_length).reshape(1, -1)
        
        # Use the environment reward as the base for task rewards
        task_rewards = np.full(reward_model.video_length, rewards[0])
        
        # Sample a clip_embedding in a realistic range (mimicking CLIP features)
        clip_embedding = np.random.normal(0, 1, reward_model.replay_buffer.clip_embeddings.shape[-1])
        
        reward_model.replay_buffer.add(
            obs=obs_seq,
            next_obs=next_obs_seq,
            action=action_seq,
            reward=rewards.reshape(1, -1),
            done=dones.reshape(1,),
            infos=[{
                "task_rewards": task_rewards,
                "clip_embedding": clip_embedding
            }]
        )
        
        obs = new_obs
        collected_steps += 1
        
        if dones[0]:
            episodes_completed += 1
            pbar.update(1)
            obs = env.reset()
    
    pbar.close()
    print(f"✅ Trajectory collection complete: {collected_steps} steps across {episodes_completed} episodes")
    
    # Rebuild the reference buffer (update reference vectors without training)
    print("🔧 Recomputing reference buffer...")
    
    # Clear GPU cache to lower OOM risk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared GPU cache")
    
    # Process the replay buffer in chunks to avoid OOM
    replay_buffer = reward_model.replay_buffer
    
    # Determine the number of available samples
    if replay_buffer.full:
        n_samples = replay_buffer.buffer_size
    else:
        n_samples = replay_buffer.pos
    
    print(f"🔢 Available samples: {n_samples}")
    
    if n_samples > 0:
        sample_rewards = []
        sample_obs_list = []
        vlm_returns_list = []
        task_rewards_list = []
        
        batch_size = 512  # batch size
        print(f"📦 Processing in batches of size {batch_size}")
        
        with torch.no_grad():
            # Batch processing to avoid OOM
            for batch_start in tqdm(range(0, n_samples, batch_size), desc="🔧 Rebuilding reference", unit="batch"):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_indices = np.arange(batch_start, batch_end)
                
                # Use ``_get_samples`` without vec-norm env to avoid normalization issues
                batch_samples = replay_buffer._get_samples(batch_indices, None)
                batch_obs = batch_samples.observations
                
                # Model predicts rewards
                pred_rewards = reward_model.model(batch_obs)
                sample_rewards.append(pred_rewards.cpu())  # move to CPU immediately to save VRAM
                
                # Collect auxiliary tensors
                sample_obs_list.append(batch_obs.cpu())
                vlm_returns_list.append(batch_samples.rewards.cpu())
                task_rewards_list.append(torch.from_numpy(replay_buffer.task_rewards[batch_indices]))
                
                # Free GPU cache
                del batch_obs, pred_rewards, batch_samples
                torch.cuda.empty_cache()
            
            if len(sample_rewards) > 0:
                # Concatenate on CPU
                sample_rewards = torch.cat(sample_rewards, dim=0)
                sample_obs = torch.cat(sample_obs_list, dim=0)
                vlm_returns = torch.cat(vlm_returns_list, dim=0)
                task_rewards = torch.cat(task_rewards_list, dim=0)
                
                print(f"📊 Combined tensor shapes: rewards={sample_rewards.shape}, obs={sample_obs.shape}")
                
                # Keep only top-k samples before moving back to GPU to limit memory usage
                if vlm_returns.shape[0] > reward_model.n_top:
                    _, sort_ind = vlm_returns.squeeze().sort()
                    top_indices = sort_ind[-reward_model.n_top:]
                    
                    sample_rewards = sample_rewards[top_indices]
                    sample_obs = sample_obs[top_indices]
                    vlm_returns = vlm_returns[top_indices]
                    task_rewards = task_rewards[top_indices]
                    
                    print(f"🎯 Selected top-{reward_model.n_top} samples, shape: {sample_rewards.shape}")
                
                # Move tensors to GPU
                reward_model.reference_rewards = sample_rewards.to(replay_buffer.device)
                reward_model.reference_vlm_returns = vlm_returns.to(replay_buffer.device)
                reward_model.reference_obs = sample_obs.to(replay_buffer.device)
                reward_model.reference_task_rewards = task_rewards.to(replay_buffer.device)
                
                # Create flattened variants
                reward_model.reference_rewards_flattened = reward_model.reference_rewards.view(-1, 1)
                reward_model.reference_task_rewards_flattened = reward_model.reference_task_rewards.view(-1, 1)
                
                print(f"✅ Completed batched processing over {n_samples} samples")
    else:
        raise RuntimeError("Insufficient samples collected; cannot rebuild reference buffer")
    
    # Validate rebuild results
    if reward_model.reference_rewards is not None:
        ref_size = reward_model.reference_rewards.shape[0]
        print("✅ Reference buffer rebuilt successfully!")
        print(f"  Reference size: {ref_size}")
        print(f"  Reference reward range: [{reward_model.reference_rewards.min():.4f}, {reward_model.reference_rewards.max():.4f}]")

        # Smoke-test ``sample_reference``
        test_refs = reward_model.sample_reference(5)
        print(f"  sample_reference output: {test_refs.squeeze()}")
        print(f"  All zeros: {torch.all(test_refs == 0).item()}")
        
        if torch.all(test_refs == 0):
            raise RuntimeError("Reference buffer still returns all zeros after rebuilding; rebuild failed")
    else:
        raise RuntimeError("Reference buffer rebuild failed; unable to create valid reference rewards")

def ensure_reference_buffer(model, env, auto_rebuild=True):
    """
    Convenience helper that makes sure the reference buffer is valid.
    Follows a fail-fast philosophy and raises if recovery is not possible.

    Args:
        model: VLMTQC model instance.
        env: Environment instance.
        auto_rebuild: Rebuild automatically when empty (default: True).

    Raises:
        RuntimeError: If the reference buffer cannot be restored.
    """
    has_reference = check_reference_buffer_status(model)
    
    if not has_reference:
        if auto_rebuild:
            print("\n⚠️ Reference buffer is empty; MVR reward predictions will be distorted!")
            print("🛠️ Launching reference buffer rebuild...")

            rebuild_reference_buffer(model, env, n_episodes=30)
            print("✅ Reference buffer rebuilt; MVR reward predictions are healthy again!")
        else:
            raise RuntimeError("Reference buffer is empty and auto rebuild is disabled; cannot execute reliable MVR analysis")
    else:
        print("✅ Reference buffer is healthy; rebuild skipped")
