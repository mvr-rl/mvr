import os
import logging

import torch
from einops import rearrange
from torch import nn
import math
import numpy as np

# from .criterions import VTC_VTM_Loss
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip_vision import clip_joint_l14, clip_joint_b16
from .viclip_text import clip_text_l14, clip_text_b16
from numpy.typing import NDArray
logger = logging.getLogger(__name__)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ViCLIP_B(nn.Module):
    """docstring for ViCLIP"""

    def __init__(self,  pretrained, target_prompts, image_width):
        super(ViCLIP_B, self).__init__()
        self.tokenizer = _Tokenizer()
        self.max_txt_l = 32
        self.vision_encoder_name = 'vit_b16'
    
        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = 8
        self.vision_encoder_drop_path_rate = 0.1
        self.vision_encoder_checkpoint_num = 24
        self.vision_width = 1024
        self.text_width = 768 
        self.embed_dim = 768 
        self.masking_prob = 0.9
        
        self.text_encoder_name = 'vit_b16'
        self.text_encoder_pretrained = False#'bert-base-uncased'
        self.text_encoder_d_model = 768

        self.text_encoder_vocab_size = 49408
        
        self.temp = nn.parameter.Parameter(torch.ones([]) * 1 / 100.0)
        self.temp_min = 1 / 100.0
        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder()
        state_dict = torch.load(pretrained, map_location='cpu')['model']
        self.load_state_dict(state_dict)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.image_width = image_width
        self.target_prompts = target_prompts
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        
        text_features = self.encode_text([target_prompts[0]]).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)  
        self.register_buffer("target", text_features)
        
    @property
    def device(self):
        return self.dummy_param.device


    def encode_vision(self, image, test=False):
        """encode image / videos as features.
        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.
        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].
        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].
        """
        device = next(self.text_encoder.parameters()).device
        text = self.text_encoder.tokenize(
            text, context_length=self.max_txt_l
        ).to(device)
        text_embeds = self.text_encoder(text)
        return text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")
            
        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        
        if encoder_name == "vit_l14":
            text_encoder = clip_text_l14(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
                tokenizer_path=None# if not 'tokenizer_path' in self.config.to_dict() else self.config.tokenizer_path
            )
        elif encoder_name == "vit_b16":
            text_encoder = clip_text_b16(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
                tokenizer_path=None# if not 'tokenizer_path' in self.config.to_dict() else self.config.tokenizer_path
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return text_encoder

    @torch.inference_mode()
    def forward(self, video_embeddings: torch.Tensor, obs=None, return_probability=False) -> NDArray:
        #current shape: (train_freq, n_env, 257, 768)
        #in other words, only works for stacked images
        sim = (video_embeddings * self.target).sum(-1).squeeze()
        return sim.cpu().numpy()

    @torch.inference_mode()
    def encode_stacked_image(self, x: NDArray, n_stack):
        video_length = 8
        # use the preprocess of CLIP
        if n_stack > 8:
            indices = np.linspace(
                0, n_stack - 1, video_length, dtype=np.long)
            x = x[:, indices]
            n_stack = 8
        # batch, n_stack, height, width, 3
        v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,1,1,3)
        v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,1,1,3)
        x = (x/255.0 - v_mean) / v_std
        x = np.transpose(x, (0, 1, 4, 2, 3))
        x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        video_embeds = self.encode_vision(x,test=True).float()
        video_embeds /= video_embeds.norm(dim=-1, keepdim=True)
        return video_embeds.unsqueeze(1)