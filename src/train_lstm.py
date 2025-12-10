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

# ---------------------------------------------
# Chargement Configs
# ---------------------------------------------

FILE_DIR = os.path.dirname(__file__)
CONFIGS_PATH = os.path.join(FILE_DIR, "configs.yaml")

parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", type=str, required=True)
parser.add_argument("--pair", type=str, default="BTC_USDT")
parser.add_argument("--run_id", type=str, required=True)
args = parser.parse_args()

PAIR = args.pair.replace("/", "_")
RUN_ID = args.run_id

BUCKET_URI = args.data_prefix.rstrip("/")
BUCKET = BUCKET_URI.replace("gs://", "").split("/")[0]

parts = BUCKET_URI.replace("gs://", "").split("/")
BASE_PATH = "/".join(parts[1:]) if len(parts) > 1 else ""

storage_client = storage.Client()

# ---------------------------------------------
# GCS Loaders
# ---------------------------------------------

def load_parquet_from_gcs(bucket_name, blob_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Missing file: gs://{bucket_name}/{blob_path}")

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)

def load_scaler_from_gcs(bucket_name, blob_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Missing scaler: gs://{bucket_name}/{blob_path}")

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)

# ---------------------------------------------
# Séquences
# ---------------------------------------------

def create_sequences(df, lookback):
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values.astype("float32")

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])  # close normalisé

    return np.array(X), np.array(y)

# ---------------------------------------------
# Modèle LSTM (anti collapse)
# ---------------------------------------------

def build_lstm(cfg, n_features):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg["lookback"], n_features)),
        tf.keras.layers.LSTM(cfg["lstm_units"], return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(cfg["lstm_units"] // 2, dropout=0.2),
        tf.keras.layers.Dense(1)
    ])

# ---------------------------------------------
# Entraînement d'une config
# ---------------------------------------------

def train_one_config(df, cfg, max_epochs):
    tf.keras.backend.clear_session()

    X, y = create_sequences(df, cfg["lookback"])
    print(f"Training config: {cfg} → X={X.shape}, y={y.shape}")

    model = build_lstm(cfg, n_features=X.shape[2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="mse")

    # split 10%
    ds_size = len(X)
    val_size = int(ds_size * 0.1)

    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val     = X[-val_size:], y[-val_size:]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=cfg["batch_size"],
        verbose=0,
        callbacks=callbacks
    )

    return model, history, (X_val, y_val)

# ---------------------------------------------
# MAE USD pour sélection modèle
# ---------------------------------------------

def compute_mae_usd(model, X_val, y_val, scaler):
    raw_pred = model.predict(X_val, verbose=0).reshape(-1)

    pred_f = np.zeros((len(raw_pred), 5))
    true_f = np.zeros((len(y_val), 5))

    pred_f[:, 3] = raw_pred
    true_f[:, 3] = y_val

    pred_usd = scaler.inverse_transform(pred_f)[:, 3]
    real_usd = scaler.inverse_transform(true_f)[:, 3]

    return float(np.mean(np.abs(pred_usd - real_usd)))

# ---------------------------------------------
# HPO complet
# ---------------------------------------------

def evaluate_all_configs(df, configs, max_epochs, scaler):
    candidates = []

    for cfg in configs:
        model, history, (X_val, y_val) = train_one_config(df, cfg, max_epochs)

        mae_usd = compute_mae_usd(model, X_val, y_val, scaler)

        candidates.append({
            "config": cfg,
            "history": [{"epoch": i+1, "val_loss": float(v)} for i, v in enumerate(history.history["val_loss"])],
            "mae_usd": mae_usd,
            "model": model
        })

    candidates.sort(key=lambda c: c["mae_usd"])
    return candidates

# ---------------------------------------------
# Load dataset + scaler
# ---------------------------------------------

train_path = f"{BASE_PATH}/{PAIR}_train_{RUN_ID}.parquet"
scaler_path = f"{BASE_PATH}/scalers/scaler_{PAIR}_{RUN_ID}.pkl"

print(f"Loading training data: gs://{BUCKET}/{train_path}")
df = load_parquet_from_gcs(BUCKET, train_path)

print(f"Loading scaler: gs://{BUCKET}/{scaler_path}")
scaler = load_scaler_from_gcs(BUCKET, scaler_path)

cols = ["open", "high", "low", "close", "volume"]
df[cols] = scaler.transform(df[cols])

# ---------------------------------------------
# Load configs.yaml
# ---------------------------------------------

with open(CONFIGS_PATH, "r") as f:
    config_yaml = yaml.safe_load(f)

configs_to_test = [
    {
        "batch_size": b,
        "lookback": l,
        "lstm_units": u,
        "learning_rate": lr
    }
    for b, l, u, lr in product(
        config_yaml["batch_sizes"],
        config_yaml["lookbacks"],
        config_yaml["lstm_units"],
        config_yaml["learning_rates"]
    )
]

max_epochs = config_yaml["max_epochs"]

# ---------------------------------------------
# Run HPO
# ---------------------------------------------

all_candidates = evaluate_all_configs(df, configs_to_test, max_epochs, scaler)
best = all_candidates[0]
model = best["model"]
history = best["history"]
best_config = best["config"]

# ---------------------------------------------
# Export (identique à ton code → Vertex AI OK)
# ---------------------------------------------

timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
model_dir = f"{BASE_PATH}/models/{PAIR}_{RUN_ID}_{timestamp}"
bucket = storage_client.bucket(BUCKET)

# SavedModel wrap
class Serve(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, best_config["lookback"], 5), dtype=tf.float32)]
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

# scaler
joblib.dump(scaler, "scaler.pkl")
bucket.blob(f"{model_dir}/scaler.pkl").upload_from_filename("scaler.pkl")
os.remove("scaler.pkl")

# train data
df.to_parquet("train_data.parquet")
bucket.blob(f"{model_dir}/train_data.parquet").upload_from_filename("train_data.parquet")
os.remove("train_data.parquet")

# metrics
metrics = {
    "run_id": RUN_ID,
    "pair": PAIR,
    "timestamp": timestamp,
    "best_config": best_config,
    "mae_usd": best["mae_usd"]
}
bucket.blob(f"{model_dir}/metrics.json").upload_from_string(json.dumps(metrics, indent=2))

# all_candidates
clean = []
for c in all_candidates:
    d = dict(c)
    d.pop("model", None)
    clean.append(d)

with open("all_candidates.json", "w") as f:
    json.dump(clean, f, indent=2)

bucket.blob(f"{model_dir}/all_candidates.json").upload_from_filename("all_candidates.json")

# keras
model.save("model.keras", save_format="keras")
bucket.blob(f"{model_dir}/model.keras").upload_from_filename("model.keras")
os.remove("model.keras")

# history
np.save("history.npy", history)
bucket.blob(f"{model_dir}/history.npy").upload_from_filename("history.npy")
os.remove("history.npy")

# model_config.json
model_cfg = {"lookback": best_config["lookback"], "n_features": 5}
with open("model_config.json", "w") as f:
    json.dump(model_cfg, f, indent=2)

bucket.blob(f"{model_dir}/model_config.json").upload_from_filename("model_config.json")
os.remove("model_config.json")

print("Saved model_config.json")
print("Training complete.")
