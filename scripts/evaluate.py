import sys
from pathlib import Path
from ultralytics import YOLO

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    # Load the FINE-TUNED model you just trained
    best_weights_path = Path(__file__).resolve().parent.parent / "models" / "bag_counter_v1" / "weights" / "best.pt"
    
    if not best_weights_path.exists():
        print("Error: Trained weights not found. Run train.py first.")
        return

    print("Loading model for evaluation...")
    model = YOLO(best_weights_path)

    # Run validation
    # This automatically uses the validation set defined in your data.yaml
    metrics = model.val()

    # Print core metrics
    print("--- Evaluation Results ---")
    print(f"mAP50-95 (Overall Accuracy): {metrics.box.map:.4f}")
    print(f"mAP50 (Standard Accuracy):   {metrics.box.map50:.4f}")
    print(f"Precision:                   {metrics.box.mp:.4f}")
    print(f"Recall:                      {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()