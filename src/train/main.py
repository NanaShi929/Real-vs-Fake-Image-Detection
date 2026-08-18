import warnings
from core_utils import setup_env, prepare_data, get_dataloaders
from model_resnet import train_resnet
from model_efficientnet import train_efficientnet
from model_vit import train_vit

warnings.filterwarnings("ignore")

def main():
    print("=== Starting Training Pipeline ===")
    
    # 1. Environment & Data Setup
    setup_env()
    TRAIN_DIR, VAL_DIR = prepare_data()
    train_loader, val_loader = get_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=256)

    # 2. Execute Model Training
    train_resnet(train_loader, val_loader)
    train_efficientnet(train_loader, val_loader)
    train_vit(train_loader, val_loader)
    
    print("\n=== Training Pipeline Complete ===")

if __name__ == "__main__":
    main()