import torch.nn as nn
from torchvision import models
from core_utils import device, run_training

def train_resnet(train_loader, val_loader):
    print("\nInitializing ResNet50...")
    resnet50 = models.resnet50(weights="DEFAULT")
    resnet50.fc = nn.Linear(resnet50.fc.in_features, 2)
    resnet50 = resnet50.to(device)
    
    run_training(resnet50, "resnet50", train_loader, val_loader)