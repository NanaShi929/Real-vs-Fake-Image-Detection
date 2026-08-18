import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from test_utils import device, load_model, CUSTOM_DIR, OUTPUT_DIR

def predict_one(model, img):
    """Predicts 'real' vs 'fake' and returns confidence score."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
    conf, pred = torch.max(probs, 1)
    label = "fake" if pred.item() == 0 else "real"
    return label, round(conf.item() * 100, 2)

def run_custom_inference():
    """Iterates through custom images and records individual model predictions."""
    if not os.path.exists(CUSTOM_DIR) or len(os.listdir(CUSTOM_DIR)) == 0:
        print(f"\nNo custom images found in {CUSTOM_DIR}. Skipping custom inference.")
        return

    print("\n=== Running Custom Test Inference ===")
    models_dict = {
        "ResNet50": load_model("resnet50"),
        "EfficientNet": load_model("efficientnet"),
        "ViT": load_model("vit")
    }
    
    rows = []
    for file in os.listdir(CUSTOM_DIR):
        path = os.path.join(CUSTOM_DIR, file)
        try:
            img = Image.open(path).convert("RGB")
            row = {"Image": file}
            
            for name, model in models_dict.items():
                pred, conf = predict_one(model, img)
                row[f"{name}_Prediction"] = pred
                row[f"{name}_Confidence"] = conf

            rows.append(row)
        except Exception as e:
            print(f"Skipped {file}: {e}")

    if rows:
        df_custom = pd.DataFrame(rows)
        print(df_custom)
        save_path = os.path.join(OUTPUT_DIR, "custom_test_all_models.csv")
        df_custom.to_csv(save_path, index=False)
        print(f"Saved custom predictions to: {save_path}")