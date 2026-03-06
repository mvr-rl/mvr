import os
from typing import List, Optional, Tuple, overload, Union, Dict, Any
import torch
import torch.distributed as dist
import numpy as np
os.environ['CURL_CA_BUNDLE'] = ''

from .XCLIP import XCLIP
from .ViCLIP import ViCLIP
from .ViCLIP_B import ViCLIP_B
from .CLIP import CLIP
from .S3DG import S3D
#from video_utils import VideoTextMatcher

from typing import Optional, Tuple

import torch

from .CLIP_reward import CLIPReward


def extract_embeddings_from_info(
    infos, reward_model, device, return_image=False):
    render_array = np.array([i["render_array"] for i in infos])
    render_array_t = torch.from_numpy(render_array).to(device)
    with torch.no_grad():
        embeddings = reward_model.encode_image(render_array_t).cpu().numpy()
    if return_image:
        return embeddings, render_array
    else:
        return embeddings
    
'''
def load_reward_model(pretrained, target_prompts, **kwargs):
    #model_name_prefix, pretrained = model_name.split("/")
    model_name_prefix = 'ViT-bigG-14'
    pretrained = model_name
    model = CLIPReward(
        model_name='ViT-bigG-14',
        pretrained=model_name,
        target_prompts_l = target_prompts,
        learned_reward_model_path=learned_reward_model_path
    )
    return model.eval()'''


def load_reward_model_from_config(config):
    if config.reward_type == "CLIP":
        return CLIPReward(
            model_name=config.model_name,
            pretrained=config.pretrained,
            target_prompts=config.target_prompts)
    elif config.reward_type == "XCLIP":
        return XCLIPReward(
            pretrained=config.pretrained,
            target_prompts=config.target_prompts)
    else:
        raise NotImplementedError
        


def compute_rewards(
    model,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    embeddings = []
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch, embeddings_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            embeddings_batch = embeddings_batch.cpu()
            embeddings.append(embeddings_batch)
            rewards[i : i + batch_size] = rewards_batch 
    return rewards, torch.cat(embeddings)


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...

    
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None, # torch.Size([B, 480, 480, 3])
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        scatter_list = []

    # torch.Size([B, 480, 480, 3]) 0-255
    worker_frames = worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)

    with torch.no_grad():
        embeddings = reward_model.embed_module(worker_frames) # or concatenate_frames(worker_frames)
        # torch.Size([B])
        rewards = reward_model(embeddings).clone()
        reward_video = reward_model.xclip.distribute_rewards(worker_frames, reward_model.target_prompts_l)
        print(reward_video)
        exit()

        # # axu rewrad
        # reward_model.adaptive = reward_model.adaptive.cuda(rank)
        # # reward_ada_old_ada_new_frames = reward_model.ada(embeddings).clone()
        # reward_model.gpt.update_adaptive(worker_frames,reward_model,rank,method='plan')
        # reward_ada = reward_model.ada(embeddings).clone()
        

    def zero_t():
        return torch.zeros_like(rewards)

    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)

def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None, # torch.Size([B, 480, 480, 3])
    worker_frames_tensor=None,
) -> Optional[torch.Tensor]:
    scatter_list = [t.cuda(rank).unsqueeze(0) for t in frames]
    worker_frames = torch.cat(scatter_list, 0)
    with torch.no_grad():
        # torch.Size([B])
        rewards = reward_model(worker_frames).clone()
    return rewards, embeddings