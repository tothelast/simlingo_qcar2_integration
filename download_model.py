#!/usr/bin/env python3
"""
Download SimLingo model from HuggingFace Hub.

This script downloads the SimLingo model and tokenizer from HuggingFace
using the provided authentication token.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, snapshot_download
import argparse

def setup_hf_token(token):
    """Setup HuggingFace authentication token."""
    try:
        login(token=token)
        print("✓ Successfully authenticated with HuggingFace")
        return True
    except Exception as e:
        print(f"✗ Failed to authenticate with HuggingFace: {e}")
        return False

def download_simlingo_model(model_name="RenzKa/simlingo", local_dir="./models"):
    """Download SimLingo model from HuggingFace Hub."""
    try:
        print(f"Downloading SimLingo model: {model_name}")
        print(f"Local directory: {local_dir}")
        
        # Create models directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("✓ Successfully downloaded SimLingo model")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download SimLingo model from HuggingFace")
    parser.add_argument(
        "--token", 
        type=str, 
        default="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",
        help="HuggingFace authentication token"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="RenzKa/simlingo",
        help="Model name on HuggingFace Hub"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./models",
        help="Local directory to save the model"
    )
    
    args = parser.parse_args()
    
    print("SimLingo Model Downloader")
    print("=" * 40)
    
    # Setup authentication
    if not setup_hf_token(args.token):
        sys.exit(1)
    
    # Download model
    if not download_simlingo_model(args.model, args.output_dir):
        sys.exit(1)
    
    print("\n✓ Model download completed successfully!")
    print(f"Model saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
