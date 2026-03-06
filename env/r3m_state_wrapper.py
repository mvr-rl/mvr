"""Wrapper that concatenates R3M image embeddings with environment state."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import torch
from voltron import instantiate_extractor, load


class R3MStateWrapper(gym.ObservationWrapper):
    """Concatenate state observations with R3M image embeddings.

    Data flow:
    raw state (state_dim,) + rendered image (H, W, 3)
    -> R3M embedding (384,) -> combined state (state_dim + 384,).
    """
    
    def __init__(
        self,
        env: gym.Env,
        model_name: str = "r-r3m-vit",
        device: str = "cuda",
        width: int = 64,
        height: int = 64,
    ):
        super().__init__(env)
        
        self.device = device
        self.width = width  
        self.height = height
        
        # Configure environment rendering parameters
        if hasattr(env, 'width'):
            env.width = width
        if hasattr(env, 'height'):
            env.height = height
            
        # Load R3M model - ResNet50 backbone yielding 384-d features
        self.r3m, self.preprocessor = load(model_name, device=device)
        self.r3m.eval()
        
        # Determine original state dimension
        state_dim = env.observation_space.shape[0]

        r3m_dim = 384

        # Define new observation space: state concatenated with R3M features
        combined_dim = state_dim + r3m_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(combined_dim,),
            dtype=np.float32
        )
        
        self.original_observation_space = env.observation_space
        self.state_dim = state_dim
        self.r3m_dim = r3m_dim
        
    def _encode_image(self, rgb_array: np.ndarray) -> np.ndarray:
        """Encode an image using R3M.

        Args:
            rgb_array: Image array (H, W, 3) uint8 in [0, 255].

        Returns:
            np.ndarray: 384-d float32 embedding.
        """
        # Convert to tensor
        from PIL import Image
        import torchvision.transforms as T
        
        pil_image = Image.fromarray(rgb_array)
        # Convert to tensor then apply R3M preprocessing
        to_tensor = T.ToTensor()
        tensor_image = to_tensor(pil_image)  # (3, H, W) [0, 1]
        image = self.preprocessor(tensor_image).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            features = self.r3m(image)  # (1, 384)
            features = features.squeeze()  # (384,) remove singleton dims

        return features.cpu().numpy().flatten()  # (384,) float32, ensure 1-D

    def observation(self, state_observation: np.ndarray) -> np.ndarray:
        """Concatenate state observation with R3M image features."""
        # Render image from environment
        rgb_array = self.render()  # (H, W, 3) uint8
        
        # Encode image via R3M
        r3m_features = self._encode_image(rgb_array)  # (384,) float32
        
        # Concatenate state and embedding
        combined_obs = np.concatenate([state_observation, r3m_features])  # (state_dim + 384,)
        
        return combined_obs.astype(np.float32)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return combined observation."""
        state_obs, info = self.env.reset(**kwargs)  # state_obs: (state_dim,)
        combined_obs = self.observation(state_obs)  # (state_dim + 384,)
        
        return combined_obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and return combined observation."""
        state_obs, reward, terminated, truncated, info = self.env.step(action)  # state_obs: (state_dim,)
        combined_obs = self.observation(state_obs)  # (state_dim + 384,)

        return combined_obs, reward, terminated, truncated, info
