import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def batch_predict(input_file, output_file, model_dir):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Reading input data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
        
    texts = df["text"].tolist()
    
    print("Running predictions...")
    predicted_sentiments = []
    confidences = []
    
    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Assuming 0 is negative, 1 is positive (standard for IMDB binary classification)
            # You might need to map to neutral based on threshold or if using 3-class model
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            max_probs = torch.max(probs, dim=-1).values.cpu().numpy()
            
            for pred, prob in zip(preds, max_probs):
                # Map to string sentiment based on standard binary classes
                # If you trained on a different dataset with neutral, adjust this mapping.
                sentiment = "positive" if pred == 1 else "negative"
                predicted_sentiments.append(sentiment)
                confidences.append(float(prob))
                
    print("Saving results...")
    df["predicted_sentiment"] = predicted_sentiments
    df["confidence"] = confidences
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to input CSV containing 'text' column")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save output CSV")
    parser.add_argument("--model-dir", type=str, default="model_output", help="Path to fine-tuned model artifacts")
    args = parser.parse_args()
    
    batch_predict(args.input_file, args.output_file, args.model_dir)
