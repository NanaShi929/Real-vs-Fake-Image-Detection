import os
import shutil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import kagglehub
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------
# PATHS & SETTINGS
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

LOG_DIR = os.path.join(MODELS_DIR, "logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure required directories exist
for p in [MODELS_DIR, LOG_DIR, OUTPUT_DIR, DATA_DIR]:
    os.makedirs(p, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_RATIO = 0.20

def setup_env():
    """Sets random seeds and configures the hardware environment."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    print("Environment setup complete. Using device:", device)

def prepare_data():
    """Downloads the dataset via Kaggle and performs train/val splitting."""
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
    
    DATASET_ROOT = dataset_path
    if not os.path.exists(os.path.join(DATASET_ROOT, "train")):
        for root, dirs, files in os.walk(DATASET_ROOT):
            if "train" in dirs and "test" in dirs:
                DATASET_ROOT = root
                break

    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    VAL_DIR   = os.path.join(DATASET_ROOT, "val")

    if not os.path.exists(os.path.join(VAL_DIR, "real")):
        print("Creating validation split...")
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
            cls_train = os.path.join(TRAIN_DIR, cls)
            cls_val   = os.path.join(VAL_DIR, cls)

            files = [f for f in os.listdir(cls_train) if os.path.isfile(os.path.join(cls_train, f))]
            _, val_files = train_test_split(files, test_size=VAL_RATIO, random_state=SEED, shuffle=True)

            for f in val_files:
                shutil.move(os.path.join(cls_train, f), os.path.join(cls_val, f))
    else:
        print("Validation already exists. Skipping split.")

    return TRAIN_DIR, VAL_DIR

def get_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=256):
    """Creates and returns PyTorch DataLoaders."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL_DIR, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def run_training(model, model_name, train_loader, val_loader, max_epochs=10, patience=2):
    """Universal PyTorch Training Loop with Early Stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_acc = 0.0
    patience_counter = 0
    log_data = []

    for epoch in range(1, max_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast("cuda"):
                    out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(f"{model_name} | Epoch {epoch} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}")
        
        log_data.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        # Save Best Model Weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{model_name}] Early stopping triggered.")
            break
            
    # Save training logs
    pd.DataFrame(log_data).to_csv(os.path.join(LOG_DIR, f"{model_name}.csv"), index=False)