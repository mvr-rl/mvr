#!/usr/bin/env python3
"""
Download ViCLIP model from ModelScope
"""

import requests
import os
from pathlib import Path

def download_viclip_model():
    """Download ViCLIP model from ModelScope mirror"""
    
    # Create directory
    model_dir = Path("ckpts/ViCLIP")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "ViCLIP-L_InternVid-FLT-10M.pth"
    
    # Check if already exists
    if model_path.exists():
        size = model_path.stat().st_size
        print(f"ViCLIP model already exists: {model_path}")
        print(f"File size: {size / (1024*1024):.1f} MB")
        return
    
    print("Downloading ViCLIP model from ModelScope...")
    
    # Download URL (ModelScope mirror for faster download in China)
    url = "https://www.modelscope.cn/models/OpenGVLab/ViCLIP/resolve/master/ViCLIP-L_InternVid-FLT-10M.pth"
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024*1024):.1f} MB")
        
        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f'\rProgress: {progress:.1f}%', end='', flush=True)
        
        print(f'\n✅ Download completed!')
        final_size = model_path.stat().st_size
        print(f"Final file size: {final_size / (1024*1024):.1f} MB")
        
        # Verify file size
        if final_size > 100 * 1024 * 1024:  # > 100MB
            print("✅ File size looks correct")
        else:
            print("❌ File size too small, download may be incomplete")
            model_path.unlink()  # Delete incomplete file
            
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial file
        raise

if __name__ == "__main__":
    download_viclip_model()
