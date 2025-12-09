import os
import io
import json
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
import joblib
import yaml

# Chargement des hyperparamètres YAML
FILE_DIR = os.path.dirname(__file__)
CONFIGS_PATH = os.path.join(FILE_DIR, "configs.yaml")

# Lecture des arguments Vertex AI
parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", type=str, required=True)
parser.add_argument("--pair", type=str, default="BTC_USDT")
parser.add_argument("--run_id", type=str, required=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lookback", type=int, default=60)
args = parser.parse_args()

# Standardisation du nom
PAIR = args.pair.replace("/", "_")
RUN_ID = args.run_id

# Préparation chemins GCS
BUCKET_URI = args.data_prefix.rstrip("/")
BUCKET = BUCKET_URI.replace("gs://", "").split("/")[0]

parts = BUCKET_URI.replace("gs://", "").split("/")
BASE_PATH = "/".join(parts[1:]) if len(parts) > 1 else ""

storage_client = storage.Client()

# Chargement Parquet depuis GCS
def load_parquet_from_gcs(bucket_name, blob_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Missing file: gs://{bucket_name}/{blob_path}")

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)

# Lecture scaler depuis GCS
def load_scaler_from_gcs(bucket_name, blob_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Missing scaler: gs://{bucket_name}/{blob_path}")

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)

# Construction des séquences LSTM
def create_sequences(df, lookback=60):
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values.astype("float32")

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])

    return np.array(X), np.array(y)

# Entraînement sur une configuration
def train_one_config(df, cfg, epochs, patience_factor):
    tf.keras.backend.clear_session()

    X, y = create_sequences(df, cfg["lookback"])
    print(f"Training config: {cfg} → X={X.shape}, y={y.shape}")

    norm = tf.keras.layers.Normalization()
    norm.adapt(X.reshape(-1, X.shape[2]))

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(2048).batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg["lookback"], X.shape[2])),
        tf.keras.layers.TimeDistributed(norm),
        tf.keras.layers.LSTM(cfg["lstm_units"]),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse")

    history_epochs = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        hist = model.fit(
            ds,
            epochs=1,
            validation_split=0.1,
            verbose=0
        )

        val_loss = hist.history["val_loss"][0]
        history_epochs.append({"epoch": epoch + 1, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_loss > patience_factor * best_val_loss:
            print(f"Pruned: val_loss {val_loss:.4f} > threshold {patience_factor * best_val_loss:.4f}")
            break

    return model, history_epochs, best_val_loss

# Recherche hyperparamètres
def evaluate_all_configs(df, configs, max_epochs, patience_factor):
    candidates = []
    for cfg in configs:
        model, hist, final_val = train_one_config(
            df, cfg, max_epochs, patience_factor
        )

        candidates.append({
            "config": cfg,
            "history": hist,
            "final_val_loss": float(final_val),
            "model": model
        })

    candidates.sort(key=lambda c: c["final_val_loss"])
    return candidates

# Chargement dataset
train_path = f"{BASE_PATH}/{PAIR}_train_{RUN_ID}.parquet"
scaler_path = f"{BASE_PATH}/scalers/scaler_{PAIR}_{RUN_ID}.pkl"

print(f"Loading training data: gs://{BUCKET}/{train_path}")
df = load_parquet_from_gcs(BUCKET, train_path)

print(f"Loading scaler: gs://{BUCKET}/{scaler_path}")
scaler = load_scaler_from_gcs(BUCKET, scaler_path)

# Application du scaler
cols = ["open", "high", "low", "close", "volume"]
df[cols] = scaler.transform(df[cols])

# Chargement configs YAML
with open(CONFIGS_PATH, "r") as f:
    config_yaml = yaml.safe_load(f)

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

# Lancement HPO
all_candidates = evaluate_all_configs(
    df,
    configs_to_test,
    max_epochs,
    patience_factor
)

best = all_candidates[0]
model = best["model"]
history = best["history"]
best_config = best["config"]

timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
model_dir = f"{BASE_PATH}/models/{PAIR}_{RUN_ID}_{timestamp}"
bucket = storage_client.bucket(BUCKET)

# Export SavedModel
class Serve(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, best_config["lookback"], 5), dtype=tf.float32)
        ]
    )
    def serving_default(self, x):
        return {"output_0": self.model(x)}

serving = Serve(model)
local_model_dir = f"saved_model_{PAIR}"
tf.saved_model.save(serving, local_model_dir)

import shutil

zip_path = shutil.make_archive(local_model_dir, 'zip', local_model_dir)
bucket.blob(f"{model_dir}/{local_model_dir}.zip").upload_from_filename(zip_path)

shutil.rmtree(local_model_dir)
os.remove(zip_path)

# Sauvegarde scaler
scaler_local_path = "scaler.pkl"
joblib.dump(scaler, scaler_local_path)
bucket.blob(f"{model_dir}/scaler.pkl").upload_from_filename(scaler_local_path)
os.remove(scaler_local_path)

# Sauvegarde dataset
train_local_path = "train_data.parquet"
df.to_parquet(train_local_path)
bucket.blob(f"{model_dir}/train_data.parquet").upload_from_filename(train_local_path)
os.remove(train_local_path)

# Sauvegarde métriques
metrics = {
    "run_id": RUN_ID,
    "pair": PAIR,
    "timestamp": timestamp,
    "epochs_ran": len(history),
    "final_val_loss": float(history[-1]["val_loss"]),
    "best_config": best_config
}
bucket.blob(f"{model_dir}/metrics.json").upload_from_string(json.dumps(metrics, indent=2))

# Sauvegarde all_candidates.json
for c in all_candidates:
    c.pop("model", None)
with open("all_candidates.json", "w") as f:
    json.dump(all_candidates, f, indent=2)
bucket.blob(f"{model_dir}/all_candidates.json").upload_from_filename("all_candidates.json")

# Sauvegarde model.keras
keras_path = "model.keras"
model.save(keras_path, save_format="keras")
bucket.blob(f"{model_dir}/model.keras").upload_from_filename(keras_path)
os.remove(keras_path)

# Sauvegarde history
np.save("history.npy", history)
bucket.blob(f"{model_dir}/history.npy").upload_from_filename("history.npy")
os.remove("history.npy")

print("Training completed successfully.")
