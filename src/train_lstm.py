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
from itertools import product
import yaml

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

def train_one_config(X, y, cfg, epochs, best_val_loss=float("inf") , patience_factor=1.2):
    """
    Train one configuration and stop early if it's clearly worse than the best so far.
    
    patience_factor: si val_loss > patience_factor * best_val_loss, on stop
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg["lookback"], X.shape[2])),
        tf.keras.layers.LSTM(cfg["lstm_units"]),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer="adam", loss="mse")

    history_epochs = []
    for epoch in range(epochs):
        hist = model.fit(
            X, y,
            epochs=1,
            batch_size=cfg["batch_size"],
            validation_split=0.1,
            verbose=0
        )
        val_loss = hist.history["val_loss"][0]
        history_epochs.append({"epoch": epoch + 1, "val_loss": val_loss})

        # Pruning: stop si beaucoup plus mauvais que le meilleur
        if val_loss > patience_factor * best_val_loss:
            print(f"Pruning config (val_loss={val_loss:.4f} > {patience_factor}*{best_val_loss:.4f}) at epoch {epoch+1}")
            break

        # Mettre à jour le meilleur val_loss pour cette config
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return model, history_epochs, best_val_loss


def successive_halving(X, y, configs, max_epochs=20, patience_factor=1.2):
    candidates = []
    best_val_loss_overall = float("inf")

    for cfg in configs:
        model, hist, best_val_loss_overall = train_one_config(
            X, y,
            cfg,
            epochs=max_epochs,
            best_val_loss=best_val_loss_overall,
            patience_factor=patience_factor
        )

        cand = {
            "config": cfg,
            "history": hist,
            "final_val_loss": hist[-1]["val_loss"],
            "model": model
        }
        candidates.append(cand)

        # Tri pour garder le meilleur val_loss global à jour
        candidates.sort(key=lambda c: c["final_val_loss"])
        best_val_loss_overall = candidates[0]["final_val_loss"]

    return candidates


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

# Lire le fichier YAML
with open("configs.yaml", "r") as f:
    config_yaml = yaml.safe_load(f)

# Générer toutes les combinaisons possibles de batch_size, lookback et lstm_units
configs_to_test = [
    {
        "batch_size": b,
        "lookback": l,
        "lstm_units": u,
        "learning_rate": config_yaml.get("learning_rate", 0.001)
    }
    for b, l, u in product(
        config_yaml["batch_sizes"],
        config_yaml["lookbacks"],
        config_yaml["lstm_units"]
    )
]

max_epochs = config_yaml.get("max_epochs", 20)
patience_factor = config_yaml.get("patience_factor", 1.2)

all_candidates = successive_halving(X, y,configs_to_test,max_epochs=max_epochs,patience_factor=patience_factor)


# garder le meilleur
best = all_candidates[0]
model = best["model"]
history = best["history"]
best_config = best["config"]

# Save the trained model to GCS
timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
model_dir = f"{BASE_PATH}/models/{PAIR}_{RUN_ID}_{timestamp}"
local_model_path = f"lstm_model_{PAIR}.keras"
model.save(local_model_path)

bucket = storage_client.bucket(BUCKET)
blob = bucket.blob(f"{model_dir}/{local_model_path}")
blob.upload_from_filename(local_model_path)
print(f"Model uploaded to gs://{BUCKET}/{model_dir}/{local_model_path}")

# Sauvegarde des metrics avec hyperparamètres
metrics = {
    "run_id": RUN_ID,
    "pair": PAIR,
    "timestamp": timestamp,
    "epochs_ran": len(history),
    "final_val_loss": float(history[-1]["val_loss"]),
    "best_config": best_config
}

metrics_blob = bucket.blob(f"{model_dir}/metrics.json")
metrics_blob.upload_from_string(json.dumps(metrics, indent=2))
print(f"Metrics uploaded: {metrics}")

# Upload all hyperparameter candidates tested
local_candidates_path = "all_candidates.json"
for c in all_candidates:
    c.pop("model", None)
with open(local_candidates_path, "w") as f:
    json.dump(all_candidates, f, indent=2)

candidates_blob = bucket.blob(f"{model_dir}/all_candidates.json")
candidates_blob.upload_from_filename(local_candidates_path)
print(f"All hyperparameter candidates uploaded to gs://{BUCKET}/{model_dir}/all_candidates.json")