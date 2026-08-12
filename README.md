# Twitter Sentiment Analysis API with BERT Fine-Tuning

This project implements a complete sentiment analysis system using a pre-trained BERT (DistilBERT) model. It includes a full MLOps pipeline covering data preprocessing, model fine-tuning with the Hugging Face library, performance evaluation, and deployment as a REST API and a Web UI.

## Model Choice
I opted to use `distilbert-base-uncased` rather than a full `bert-base-uncased` model. DistilBERT provides ~97% of BERT's performance while being 40% smaller and 60% faster. This significantly reduces Docker image sizes, memory consumption, and API latency, which is crucial for building responsive web applications.

## Prerequisites
- Python 3.10+ (for local training)
- Docker & Docker Compose (for containerized deployment)

## Setup Instructions

### 1. Model Training (Required Before Deployment)
The FastAPI Docker image expects the fine-tuned model artifacts to be present in the `model_output/` directory.

To generate these artifacts, you can run the training script locally. We recommend setting up a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install required dependencies for training
pip install pandas transformers datasets evaluate torch scikit-learn

# Preprocess the data (using IMDB dataset)
python scripts/preprocess.py

# Fine-tune the model
# This will save artifacts to model_output/ and metrics to results/
python scripts/train.py
```

*Note: For testing purposes, you can run `python scripts/preprocess.py --num-samples 20` to train on a very small subset of data quickly.*

### 2. Environment Configuration
Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```
Modify any variables in `.env` if necessary.

### 3. Deploying the Application
Once the model artifacts are present in `model_output/`, build and start the Docker containers:

```bash
docker-compose up --build -d
```

Check the status of the containers to ensure they are healthy:
```bash
docker ps
```

### 4. Accessing the Application
- **Web UI:** Open [http://localhost:8501](http://localhost:8501) in your browser.
- **API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## API Usage Guide

### Health Check
```bash
curl http://localhost:8000/health
```
**Response:** `{"status": "ok"}`

### Sentiment Prediction
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this new feature!"}'
```
**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.99
}
```

## Batch Prediction
You can also run predictions on a CSV file using the standalone script:

```bash
python scripts/batch_predict.py --input-file data/unseen/data.csv --output-file results/predictions.csv
```
Make sure the input CSV has a `text` column. The script will output a CSV with `predicted_sentiment` and `confidence` columns appended.
