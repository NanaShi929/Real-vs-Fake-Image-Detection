import timm
from core_utils import device, run_training

def train_vit(train_loader, val_loader):
    print("\nInitializing ViT...")
    vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    vit = vit.to(device)
    
    run_training(vit, "vit", train_loader, val_loader)