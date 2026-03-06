#!/usr/bin/env python3
"""
Download ViCLIP-B model by cloning from Hugging Face and converting to pth
"""

import subprocess
import os
import torch
from pathlib import Path

def download_viclip_model():
    """Download ViCLIP-B model from Hugging Face and convert to pth"""
    
    # Create directory
    model_dir = Path("ckpts/ViCLIP")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "ViCLIP-B-16-hf.pth"
    hf_dir = Path("ckpts/ViCLIP-B-16-hf")
    
    # Check if already exists
    if model_path.exists():
        size = model_path.stat().st_size
        print(f"ViCLIP model already exists: {model_path}")
        print(f"File size: {size / (1024*1024):.1f} MB")
        return
    
    print("Downloading ViCLIP-B model from Hugging Face...")
    
    try:
        # Clone from Hugging Face
        if not hf_dir.exists():
            print("Cloning repository...")
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/OpenGVLab/ViCLIP-B-16-hf",
                str(hf_dir)
            ], check=True)
            
            print("Installing git lfs and downloading model files...")
            subprocess.run(["git", "lfs", "install"], check=True, cwd=str(hf_dir))
            subprocess.run(["git", "lfs", "pull"], check=True, cwd=str(hf_dir))
        
        # Convert safetensors to pth
        safetensors_path = hf_dir / "model.safetensors"
        
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Model file not found: {safetensors_path}")
        
        print("Converting safetensors to pth format...")
        from safetensors.torch import load_file
        
        # Load and convert
        state_dict = load_file(safetensors_path)
        # Wrap in 'model' key as expected by the ViCLIP loader
        model_dict = {'model': state_dict}
        torch.save(model_dict, model_path)
        
        print(f'✅ Download and conversion completed!')
        final_size = model_path.stat().st_size
        print(f"Final file size: {final_size / (1024*1024):.1f} MB")
        
        # Verify file size
        if final_size > 100 * 1024 * 1024:  # > 100MB
            print("✅ File size looks correct")
        else:
            print("❌ File size too small, conversion may be incomplete")
            model_path.unlink()
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        raise

if __name__ == "__main__":
    download_viclip_model()
