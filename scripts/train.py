import argparse
import os
import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    prec = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    rec = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

def train(data_dir, output_dir, results_dir, model_name, epochs, batch_size, lr):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    
    print(f"Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
        
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./trainer_logs",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    # Save metrics.json
    final_metrics = {
        "accuracy": metrics.get("eval_accuracy", 0.0),
        "precision": metrics.get("eval_precision", 0.0),
        "recall": metrics.get("eval_recall", 0.0),
        "f1_score": metrics.get("eval_f1_score", 0.0)
    }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    # Save run_summary.json
    run_summary = {
        "hyperparameters": {
            "model_name": model_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_epochs": epochs
        },
        "final_metrics": final_metrics
    }
    
    summary_path = os.path.join(results_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=4)
        
    print("Saving model artifacts...")
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    print(f"Metrics saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="model_output")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    train(
        args.data_dir, 
        args.output_dir, 
        args.results_dir,
        args.model_name,
        args.epochs,
        args.batch_size,
        args.lr
    )
