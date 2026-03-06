import numpy as np
import torch
import time  # Import time for timing
from transformers import XCLIPProcessor, XCLIPModel
from torch import nn 
from numpy.typing import NDArray
from einops import rearrange
from .CLIP_reward import image_tensor_transform

class XCLIP(nn.Module):
    def __init__(self, pretrained, target_prompts, image_width):
        """
        Initialize the X-CLIP model and processor.
        """
        super().__init__()
        self.model = XCLIPModel.from_pretrained(pretrained)
        self.processor = XCLIPProcessor.from_pretrained(pretrained)
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

    def sample_frames(self, video, sample_len=8):
        """
        Sample frames from the video.
        
        Args:
            video (tensor): video. # shape [num_frames, 224, 224, 3]
            sample_len (int): Number of frames to sample.
        
        Returns:
            np.ndarray: Array of sampled frames. # shape [sample_len, 224, 224, 3]
        """
        num_frames = video.shape[0]
        
        if num_frames <= sample_len:
            # If the video has fewer frames than sample_len, just return all frames
            sampled_frames = video
        else:
            # Uniformly sample frame indices
            indices = torch.linspace(0, num_frames - 1, sample_len, dtype=torch.long).to(video.device)
            sampled_frames = video[indices]
        
        # print(f"Frame sampling time: {time.time() - start_time:.4f} seconds")
        return sampled_frames

    def run_inference(self, video_frames, texts):
        """
        Run inference on the video and texts using the X-CLIP model.
        
        Args:
            video_frames (np.ndarray): Array of sampled video frames.
            texts (list): List of text descriptions to match with the video.
        
        Returns:
            torch.Tensor: Softmax probabilities of text matching the video.
        """
        start_time = time.time()
        inputs = self.processor(text=texts, videos=list(video_frames), return_tensors="pt", padding=True)
        
        # Move inputs to GPU
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        # print(f"Preprocessing time: {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print(f"Inference time: {time.time() - start_time:.4f} seconds")
        
        start_time = time.time()
        probs = outputs.logits_per_video.softmax(dim=1)
        # print(f"Softmax computation time: {time.time() - start_time:.4f} seconds")
        
        return probs

    @torch.inference_mode()
    def encode_text(self, x):
        x = self.processor(text=x, return_tensors="pt", padding=True)
        x = x.to(self.device)
        x = self.model.get_text_features(**x)
        return x

    @torch.inference_mode()
    def encode_image(self, x: NDArray):
        raise NotImplementedError

    @torch.inference_mode()
    def encode_stacked_image(self, x: NDArray, n_stack):
        video_length = 8
        # For xclip, x should have shape  n_env, height, width, 3 * n_stack
        # use the preprocess of CLIP
        x = torch.from_numpy(x).to(self.device)
        batch_size = x.shape[0]
        if n_stack > 8:
            indices = torch.linspace(
                0, n_stack - 1, video_length, dtype=torch.long, device=x.device)
            x = x[:, indices]
            n_stack = 8
        # change to n_env * n_stack, 3,  height, width
        x = rearrange(x, "n_envs n_stack height width n_channel-> (n_envs n_stack) n_channel height width")
        
        pixel_values = self.transform(x)
        vision_outputs = self.model.vision_model(pixel_values)
        
        video_embeds = vision_outputs[1]
        video_embeds = self.model.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, n_stack, -1)

        mit_outputs = self.model.mit(
            cls_features,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.return_dict,
        )
        video_embeds = mit_outputs[1]

        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.model.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.model.prompts_visual_projection
        img_features = img_features.view(batch_size, n_stack, -1, video_embeds.shape[-1])
        img_features = img_features.mean(dim=1, keepdim=False)
        #hidden_states = hidden_states.view(n_images, -1)
        return torch.cat([img_features, video_embeds.unsqueeze(1)], 1)
    
    @torch.inference_mode()
    def forward(self, image_embeddings: torch.Tensor, obs=None, return_probability=False) -> NDArray:
        #current shape: (train_freq, n_env, 257, 768)
        #in other words, only works for stacked images
        batch_size, n_env = image_embeddings.shape[:2]
        img_features, video_embeds = torch.split(image_embeddings, [256, 1], dim=-2)
        video_embeds = video_embeds.squeeze(-2)
        img_features = rearrange(img_features, "train_freq n_env ... -> (train_freq n_env) ...")
        video_embeds = rearrange(video_embeds, "train_freq n_env ... -> (train_freq n_env) ...")
        
        text_embeds = self.target.unsqueeze(0).expand(batch_size * n_env, -1, -1)
        text_embeds = text_embeds + self.model.prompts_generator(text_embeds, img_features)
        # normalized features
        video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        sim = torch.einsum("bd,bkd->bk", video_embeds, text_embeds).squeeze()
        return sim.cpu().numpy()
