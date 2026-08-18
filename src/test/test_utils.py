import os
import torch
import torch.nn as nn
from torchvision import models
import timm

# ---------------------------------------------------------------
# PATHS & SETTINGS
# ---------------------------------------------------------------
# Resolves root path: mlops-project/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CUSTOM_DIR = os.path.join(DATA_DIR, "custom_test")

CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
LOG_DIR = os.path.join(MODELS_DIR, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CUSTOM_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(name):
    """Loads the model architecture and loads saved checkpoint weights."""
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "vit":
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)

    m = m.to(device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")
    
    if os.path.exists(checkpoint_path):
        m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
    m.eval()
    return m