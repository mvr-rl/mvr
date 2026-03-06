import numpy as np
import torch
import time  # Import time for timing
from transformers import XCLIPProcessor, XCLIPModel
from torch import nn 
from numpy.typing import NDArray
from einops import rearrange
from .CLIP_reward import image_tensor_transform
from .s3dg import S3D as RoboS3D
import os

class S3D(nn.Module):
    def __init__(self, pretrained, target_prompts, image_width):
        """
        Initialize the X-CLIP model and processor.
        """
        super().__init__()
        self.model = RoboS3D(pretrained)
        # Determine weight path based on whether pretrained is a filepath
        if isinstance(pretrained, str):
            weight_dir = os.path.dirname(pretrained)  # Derived directory of the pretrained argument
            weight_path = os.path.join(weight_dir, 's3d_howto100m.pth')  # Build the absolute weight path
        else:
            weight_path = 's3d_howto100m.pth'  # Default to loading from current directory
        # Load the model weights
        self.model.load_state_dict(torch.load(weight_path))
        # Evaluation mode
        self.model = self.model.eval()
        self.transform = image_tensor_transform(image_width)
        self.target_prompts = target_prompts
        self.dummy_param = nn.Parameter(torch.empty(0))
        target = self.encode_text([target_prompts[0]])
            #"a humanoid robot making a run posture"
        self.register_buffer("target", target)
        # Need to retrive image size and embedding dim from model
        #self.image_embedding_dim = (257, 768)
        
    @property
    def device(self):
        return self.dummy_param.device

    @torch.inference_mode()
    def encode_text(self, x):
        # x = x.to(self.device)
        x = self.model.text_module(x)
        x = x['text_embedding']
        return x

    @torch.inference_mode()
    def encode_image(self, x: NDArray):
        raise NotImplementedError

    @torch.inference_mode()
    def encode_stacked_image(self, x: NDArray, n_stack):
        video_length = 8
        # For xclip, x should have shape  n_env, height, width, 3 * n_stack
        # use the preprocess of CLIP
        x = torch.from_numpy(x).to(self.device) # [32, 8, 224, 224, 3]
        batch_size = x.shape[0]
        if n_stack > 8:
            indices = torch.linspace(
                0, n_stack - 1, video_length, dtype=torch.long, device=x.device)
            x = x[:, indices]
            n_stack = 8
        # change to n_env * n_stack, 3,  height, width
        x = rearrange(x, "n_envs n_stack height width n_channel-> n_envs n_channel n_stack height width") # torch.Size([32, 3, 8, 224, 224])
        
        # pixel_values = self.transform(x)
        # print(pixel_values.shape)
        video_output = self.model(x.float())
        x = video_output['video_embedding'].unsqueeze(1) # torch.Size([32, 512])
        # pixel_values = pixel_values.to(self.device)
        return x
    
    @torch.inference_mode()
    def forward(self, image_embeddings: torch.Tensor, obs=None, return_probability=False) -> NDArray:
        target = self.target.unsqueeze(0) # torch.Size([1, 1, 512])
        print(image_embeddings.shape) # torch.Size([32, 1, 512])
        print(target.shape) # torch.Size([1, 512])
        # sim = torch.matmul(target, image_embeddings)
        # Step 2: Compute dot product
        dot_product = (image_embeddings * target).sum(dim=-1) # [batch_size, 1]

        # Step 3: Compute norm
        image_norm = torch.norm(image_embeddings, dim=-1) # [batch_size, 1]
        target_norm = torch.norm(target, dim=-1) # Scalar

        # Step 4: Calculate cosine similarity
        sim = dot_product / (image_norm * target_norm + 1e-8) # torch.Size([32, 1])

        return sim.squeeze(1).cpu().numpy() # torch.Size([32,])
