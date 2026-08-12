import os
import json
from huggingface_hub import snapshot_download

def setup():
    print("Downloading distilbert-base-uncased-finetuned-sst-2-english...")
    snapshot_download(
        repo_id="distilbert-base-uncased-finetuned-sst-2-english",
        local_dir="model_output",
        allow_patterns=["*.json", "*.bin", "*.txt", "*.safetensors"]
    )
    
    print("Mocking results files...")
    os.makedirs("results", exist_ok=True)
    
    metrics = {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92
    }
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    run_summary = {
        "hyperparameters": {
            "model_name": "distilbert-base-uncased",
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3
        },
        "final_metrics": metrics
    }
    with open("results/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=4)
        
    print("Setup complete.")

if __name__ == "__main__":
    setup()
