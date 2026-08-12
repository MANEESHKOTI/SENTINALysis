import argparse
import os
import pandas as pd
from datasets import load_dataset
import re

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters but keep punctuation like .,!?
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']', '', text)
    # Convert to lowercase
    text = text.lower().strip()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess(output_dir, num_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading IMDB dataset...")
    # Load dataset
    dataset = load_dataset("imdb")
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Optionally subset for faster testing/demo
    if num_samples is not None:
        print(f"Subsetting data to {num_samples} samples per split for demonstration.")
        # Balance classes if possible, but for simplicity just take first N
        # We'll take top N/2 pos and N/2 neg if possible, or just shuffle and select
        train_data = train_data.shuffle(seed=42).select(range(num_samples))
        test_data = test_data.shuffle(seed=42).select(range(num_samples))
        
    print("Cleaning and converting to DataFrames...")
    # Convert to pandas
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    
    # Clean text
    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)
    
    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Saved preprocessed data to:\n- {train_path} ({len(df_train)} samples)\n- {test_path} ({len(df_test)} samples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IMDB dataset for sentiment analysis")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save preprocessed CSVs")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to subset (for faster testing)")
    args = parser.parse_args()
    
    preprocess(args.output_dir, args.num_samples)
