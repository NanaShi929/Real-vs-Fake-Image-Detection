import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import kagglehub

from test_utils import device, load_model, LOG_DIR, OUTPUT_DIR, CHECKPOINT_DIR

def prepare_test_loader(batch_size=256):
    """Fetches the dataset and returns the test DataLoader."""
    dataset_path = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")
    DATASET_ROOT = dataset_path
    if not os.path.exists(os.path.join(DATASET_ROOT, "train")):
        for root, dirs, files in os.walk(DATASET_ROOT):
            if "train" in dirs and "test" in dirs:
                DATASET_ROOT = root
                break

    TEST_DIR = os.path.join(DATASET_ROOT, "test")
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

def calculate_metrics(model_name, test_loader):
    """Runs evaluation for a model on test dataset and saves plots."""
    print(f"[{model_name.upper()}] Evaluating test set metrics...")
    model = load_model(model_name)
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out = model(x)
            probs = F.softmax(out, dim=1)
            _, pred = torch.max(out, 1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    params = sum(p.numel() for p in model.parameters())

    log_path = os.path.join(LOG_DIR, f"{model_name}.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        epochs_used = len(log)
        best_val_acc = log["val_acc"].max()
        final_valloss = log["val_loss"].iloc[-1]

        # Save Loss Curve
        plt.figure(figsize=(6, 4))
        plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
        plt.plot(log["epoch"], log["val_loss"], label="Val Loss")
        plt.title(f"{model_name} Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_loss.png"))
        plt.close()

        # Save Accuracy Curve
        plt.figure(figsize=(6, 4))
        plt.plot(log["epoch"], log["val_acc"], label="Val Accuracy")
        plt.title(f"{model_name} Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_acc.png"))
        plt.close()
    else:
        epochs_used, best_val_acc, final_valloss = 0, 0, 0

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_cm.png"))
    plt.close()

    return [
        model_name,
        round(acc, 4),
        round(prec, 4),
        round(rec, 4),
        round(f1, 4),
        round(auc, 4),
        round(ll, 4),
        params,
        epochs_used,
        round(best_val_acc, 4),
        round(final_valloss, 4)
    ]

def run_test_evaluation():
    """Runs test evaluation across all models with available checkpoints."""
    test_loader = prepare_test_loader()
    results = []
    
    for m in ["resnet50", "efficientnet", "vit"]:
        if os.path.exists(os.path.join(CHECKPOINT_DIR, f"{m}_best.pth")):
            results.append(calculate_metrics(m, test_loader))

    if results:
        df = pd.DataFrame(
            results,
            columns=[
                "Model", "Accuracy", "Precision", "Recall", "F1 Score",
                "ROC-AUC", "Log Loss", "Parameters", "Epochs Used",
                "Best Val Accuracy", "Final Val Loss"
            ]
        )
        print("\n=== Test Set Metrics ===")
        print(df)
        save_path = os.path.join(OUTPUT_DIR, "final_professional_comparison.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved benchmark to: {save_path}")