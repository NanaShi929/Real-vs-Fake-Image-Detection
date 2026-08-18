import warnings
from test_metrics import run_test_evaluation
from test_custom import run_custom_inference

warnings.filterwarnings("ignore")

def main():
    print("=== Starting Test & Evaluation Pipeline ===")
    
    # 1. Evaluate models on the Kaggle Test Dataset
    run_test_evaluation()
    
    # 2. Run inference on custom local images
    run_custom_inference()
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()