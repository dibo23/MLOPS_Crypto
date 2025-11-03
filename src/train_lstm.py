import os
import io
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
import joblib

# Parse arguments (passed from Vertex AI or CLI)
parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", type=str, required=True)
parser.add_argument("--pair", type=str, default="BTC_USDT")
parser.add_argument("--run_id", type=str, required=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lookback", type=int, default=60)
args = parser.parse_args()

# Define bucket URI and path components
BUCKET_URI = args.data_prefix.rstrip("/")  # e.g. gs://bucket-crypto-data/ohlcv-data
BUCKET = BUCKET_URI.replace("gs://", "").split("/")[0]
BASE_PATH = "/".join(BUCKET_URI.replace("gs://", "").split("/")[1:])

PAIR = args.pair
RUN_ID = args.run_id
LOOKBACK = args.lookback

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Helper function to load Parquet file from Google Cloud Storage
def load_parquet_from_gcs(bucket_name, blob_path):
    """Download a Parquet file from GCS and load it into a DataFrame."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)

# Helper function to load a scaler (saved with joblib) from GCS
def load_scaler_from_gcs(bucket_name, blob_path):
    """Download a scaler file from GCS and load it using joblib."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)

# Helper function to create LSTM sequences
def create_sequences(df, lookback=60):
    """Generate sequences of data for training LSTM model."""
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])  # 'close' column as target
    return np.array(X), np.array(y)

# Load training data and scaler from GCS
train_path = f"{BASE_PATH}/{PAIR}_train_{RUN_ID}.parquet"
scaler_path = f"{BASE_PATH}/scalers/scaler_{PAIR}_{RUN_ID}.pkl"

print(f"Loading training data from gs://{BUCKET}/{train_path}")
df = load_parquet_from_gcs(BUCKET, train_path)

print(f"Loading scaler from gs://{BUCKET}/{scaler_path}")
scaler = load_scaler_from_gcs(BUCKET, scaler_path)

# Create sequences for LSTM training
X, y = create_sequences(df, LOOKBACK)
print(f"Sequences ready: X={X.shape}, y={y.shape}")

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(LOOKBACK, X.shape[2])),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(1)
])

# Compile the model with Adam optimizer and MSE loss
model.compile(optimizer="adam", loss="mse")
print(model.summary())

# Train the model
history = model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1, verbose=1)

# Save the trained model to GCS
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
model_dir = f"{BASE_PATH}/models/{PAIR}_{RUN_ID}_{timestamp}"
local_model_path = f"lstm_model_{PAIR}.keras"
model.save(local_model_path)

bucket = storage_client.bucket(BUCKET)
blob = bucket.blob(f"{model_dir}/{local_model_path}")
blob.upload_from_filename(local_model_path)
print(f"Model uploaded to gs://{BUCKET}/{model_dir}/{local_model_path}")

# Save training metrics to GCS
metrics = {
    "run_id": RUN_ID,
    "pair": PAIR,
    "timestamp": timestamp,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lookback": args.lookback,
    "loss": float(history.history["loss"][-1]),
    "val_loss": float(history.history["val_loss"][-1])
}

metrics_blob = bucket.blob(f"{model_dir}/metrics.json")
metrics_blob.upload_from_string(json.dumps(metrics, indent=2))
print(f"Metrics uploaded: {metrics}")
