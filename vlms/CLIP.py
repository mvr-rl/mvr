import numpy as np
import torch
import time  # Import time for timing
from transformers import XCLIPProcessor, XCLIPModel
from torch import nn 
from numpy.typing import NDArray
from einops import rearrange
from .CLIP_reward import image_tensor_transform
import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from typing import Any, Dict, List, Optional, Tuple, Union, overload

class CLIP(nn.Module):
    def __init__(self, pretrained, target_prompts, image_width, model_name='ViT-H-14') -> None:
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
        self.model,_,self.processor = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained)
        # self.processor = XCLIPProcessor.from_pretrained(pretrained)
        self.transform = image_tensor_transform(image_width)
        self.target_prompts = target_prompts
        self.dummy_param = nn.Parameter(torch.empty(0))
        #target = self.encode_text(target_prompts).mean(dim=0, keepdim=True)
        target = self.encode_text([target_prompts[0]])
        self.register_buffer("target", target)
    @property
    def device(self):
        return self.dummy_param.device
    
    @torch.inference_mode()
    def forward(self, image_embeddings: torch.Tensor, obs: torch.Tensor = None) -> torch.Tensor:
        # image_embeddings: [batch_size, 1, feature_dim]
        # self.target: [feature_dim]

        # Step 1: Expand the dimensions of the target vector to match image_embeddings
        target = self.target.unsqueeze(0).unsqueeze(1) # [1, 1, feature_dim]

        # Step 2: Compute dot product
        dot_product = (image_embeddings * target).sum(dim=-1) # [batch_size, 1]

        # Step 3: Compute norm
        image_norm = torch.norm(image_embeddings, dim=-1) # [batch_size, 1]
        target_norm = torch.norm(target, dim=-1) # Scalar

        # Step 4: Calculate cosine similarity
        cosine_similarity = dot_product / (image_norm * target_norm + 1e-8) # [batch_size, 1]

        return cosine_similarity.squeeze(1).cpu().numpy() # Returns a NumPy array of [batch_size]

    @torch.inference_mode()
    def encode_text(self, x: List[str]) -> torch.Tensor:
        """Embed a list of prompts."""
        tokens = open_clip.tokenize(x)
        x = self.model.encode_text(tokens).float()
        x = x / x.norm(dim=-1, keepdim=True)
        x = x.mean(0)
        return x
    
    @torch.inference_mode()
    def encode_image(self, x: np.typing.NDArray):
        # for clip, x should have shape n_env, height, width, 3
        # x = rearrange(
        #     x, "n_envs height width n_channel -> n_envs n_channel height width")
        # x = torch.from_numpy(x).to(self.device)
        x = self.transform(x)
        x = self.model.encode_image(x, normalize=True)
        return x
    
    @torch.inference_mode()
    def encode_stacked_image(self, x: NDArray, n_stack):
        video_length = 8
        # x have shape  n_env, n_stack, height, width, 3
        # use the preprocess of CLIP
        x = torch.from_numpy(x).to(self.device) # torch.Size([4, 8, 224, 224, 3])
        batch_size = x.shape[0]
        if n_stack > 8:
            indices = torch.linspace(
                0, n_stack - 1, video_length, dtype=torch.long, device=x.device)
            x = x[:, indices]
            n_stack = 8
        x = x[:,0,:,:,:]
        # change to n_env * n_stack, 3,  height, width
        x = rearrange(x, "n_envs height width n_channel-> n_envs n_channel height width")
        # x = rearrange(x, "n_envs n_stack height width n_channel-> (n_envs n_stack) n_channel height width")
        # print(x.shape)
        
        pixel_values = self.transform(x) # torch.Size([32, 3, 224, 224])
        image_embeddings = self.model.encode_image(pixel_values, normalize=True).unsqueeze(1) # torch.Size([8,(1) 1024]) 2,(1)1024
        return image_embeddings
