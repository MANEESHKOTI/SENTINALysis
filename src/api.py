from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="API for classifying text sentiment using a fine-tuned BERT model.",
    version="1.0.0"
)

# Global variables to hold model and tokenizer
model = None
tokenizer = None
device = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    model_path = os.getenv("MODEL_PATH", "/app/model_output")
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model directory not found at {model_path}. Please ensure the model is trained and artifacts are present.")
        
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint to ensure API is running."""
    # Also verify model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """Predict the sentiment of the given text."""
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        # Tokenize input
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Extract prediction (0: negative, 1: positive)
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs, dim=-1).values.item()
            
            sentiment = "positive" if pred_idx == 1 else "negative"
            
            return PredictionResponse(sentiment=sentiment, confidence=confidence)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
