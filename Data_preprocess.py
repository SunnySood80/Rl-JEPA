import os, torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # syncs CUDA so you see the true Python stack
torch.autograd.set_detect_anomaly(True)   # pinpoints the backward op

import urllib.request
import zipfile
import socket
from datasets import load_dataset

# Check if datasets exist

imagenet_path = os.path.join('.', 'imagenet-100')
ade_path = os.path.join('.', 'ADEChallengeData2016')

print("Loading datasets...")

# Load ImageNet-100 and save to visible folder
if os.path.exists(imagenet_path):
    print("ImageNet-100 already exists. Skipping download.")
else:
    print("Downloading ImageNet-100...")
    try:
        img100 = load_dataset("clane9/imagenet-100")
        img100.save_to_disk(imagenet_path)
        print("ImageNet-100 saved to ./imagenet-100/")
        print(f"Train samples: {len(img100['train'])}")
        print(f"Val samples: {len(img100['validation'])}")
    except Exception as e:
        print(f"ImageNet-100 failed: {e}")

# Download ADE20K manually if not exists
if os.path.exists(ade_path):
    print("ADE20K already exists. Skipping download.")
else:
    print("Downloading ADE20K...")
    try:
        socket.setdefaulttimeout(60)
        url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        zip_path = "ADEChallengeData2016.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_path)
        print("ADE20K downloaded")
    except Exception as e:
        print(f"ADE20K download failed: {e}")

print("Ready.")