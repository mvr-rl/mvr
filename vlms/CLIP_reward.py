import os
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize)

import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def image_tensor_transform(
    image_size: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that
        # Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)

    def convert_from_uint8_to_float(image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            return image.to(torch.float32) / 255.0
        else:
            return image

    return Compose(
        [
            convert_from_uint8_to_float,
            Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            normalize,
        ]
    )


class CLIPReward(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        target_prompts: str,
    ) -> None:
        """

        Args:
            model (str): CLIP model.
            device (str): Device to use.
            alpha (float, optional): Coeefficient of projection.
            target_prompts (torch.Tensor): Tokenized prompts describing
                the target state.
            baseline_prompts (torch.Tensor): Tokenized prompts describing
                the baseline state.
        """
        super().__init__()
        self.model = open_clip.create_model(
            model_name=model_name, pretrained=pretrained)
        if isinstance(self.clip_model.visual.image_size, int):
            image_size = self.clip_model.visual.image_size
        else:
            image_size = self.clip_model.visual.image_size[0]
        self.transform = image_tensor_transform(image_size)
        self.target_prompts = target_prompts
        self.dummy_param = nn.Parameter(torch.empty(0))
        #target = self.encode_text(target_prompts).mean(dim=0, keepdim=True)
        target = self.encode_text([target_prompts[0], "a humanoid robot standing", "a humanoid robot falling down"])
        behavior_values = np.array([10, -5, -10])
        self.register_buffer("target", target)
        self.register_buffer(
            "behavior_values", torch.from_numpy(behavior_values))
        self.image_embedding_dim = (257, 768)

    @property
    def device(self):
        return self.dummy_param.device
    
    @torch.inference_mode()
    def forward(self, image_embeddings: torch.Tensor, obs: torch.Tensor=None) -> torch.Tensor:
        # x 
        if self.learned_reward_model:
            return self.learned_reward_model(image_embeddings, obs)
            # y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2 # cancel projection
            #y = 1 - (torch.norm((x - self.target), dim=-1) ** 2) / 2
        else:
            y = (image_embeddings * self.target).sum(1)
            return y

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        """Embed a list of prompts."""
        tokens = open_clip.tokenize(x)
        x = self.clip_model.encode_text(tokens).float()
        x = x / x.norm(dim=-1, keepdim=True)
        x = x.mean(0)
        return x
    
    @torch.inference_mode()
    def encode_image(self, x: np.typing.NDArray):
        # for clip, x should have shape n_env, height, width, 3
        x = rearrange(
            x, "n_envs height width n_channel -> n_envs n_channel height width")
        x = torch.from_numpy(x).to(self.device)
        x = self.transform(x)
        x = self.clip_model.encode_image(x, normalize=True)
        return x
    
    @torch.inference_mode()
    def encode_stacked_image(self, x: np.typing.NDArray, n_stack):
        raise NotImplementedError