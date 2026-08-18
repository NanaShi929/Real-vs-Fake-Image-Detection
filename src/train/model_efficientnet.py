import torch.nn as nn
from torchvision import models
from core_utils import device, run_training

def train_efficientnet(train_loader, val_loader):
    print("\nInitializing EfficientNetB0...")
    efficientnet = models.efficientnet_b0(weights="DEFAULT")
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
    efficientnet = efficientnet.to(device)
    
    run_training(efficientnet, "efficientnet", train_loader, val_loader)