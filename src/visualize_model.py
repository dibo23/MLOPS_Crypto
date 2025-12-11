import os
import argparse
import json
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import tempfile
import zipfile
import uuid
from google.cloud import storage

# Crée un dossier si nécessaire
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Télécharge un fichier GCS et renvoie ses bytes
def load_from_gcs(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_bytes()

# Dézippe un SavedModel et retourne un vrai modèle Keras (compatible Keras 3)
def load_savedmodel_zip(tmp_file):
    extract_dir = os.path.join(tempfile.mkdtemp(prefix="savedmodel_"), "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(tmp_file, "r") as z:
        z.extractall(extract_dir)

    return tf.keras.layers.TFSMLayer(extract_dir, call_endpoint="serving_default")

# Gestion robuste des formats modèle exportés par Vertex AI
def load_model_from_gcs_safely(bucket_name, blob_name):
    print(f"Downloading model from: {blob_name}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp_dir = tempfile.mkdtemp(prefix="model_" + str(uuid.uuid4())[:8] + "_")
    tmp_file = os.path.join(tmp_dir, os.path.basename(blob_name))
    blob.download_to_filename(tmp_file)

    if blob_name.endswith(".zip"):
        print("→ Loading SavedModel ZIP via TFSMLayer")
        return load_savedmodel_zip(tmp_file)

    if blob_name.endswith(".keras"):
        print("→ Loading .keras (Vertex) as SavedModel ZIP")
        return load_savedmodel_zip(tmp_file)

    if blob_name.endswith(".h5"):
        print("→ Loading native H5 Keras model")
        return tf.keras.models.load_model(tmp_file)

    raise ValueError(f"Unsupported model format for file: {blob_name}")

# Recherche du dossier modèle (avec RUN_ID facultatif)
def get_model_folder(bucket_name, base_path, run_id=None):
    client = storage.Client()
    prefix = f"{base_path}/models/"
    folders = set()

    for blob in client.list_blobs(bucket_name, prefix=prefix):
        parts = blob.name.split("/")
        if len(parts) >= 4:
            folders.add("/".join(parts[:3]) + "/")

    if not folders:
        raise ValueError(f"Aucun dossier modèle trouvé sous {prefix}")

    folders = sorted(folders)

    if run_id is None:
        print("Latest model folder:", folders[-1].rstrip("/"))
        return folders[-1].rstrip("/")

    for f in folders:
        if run_id in f:
            print("Selected model folder:", f.rstrip("/"))
            return f.rstrip("/")

    raise ValueError(f"Aucun modèle trouvé contenant RUN_ID={run_id}")

# Recherche du fichier modèle optimal (ZIP → .keras → .h5)
def find_model_file(bucket_name, model_folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=model_folder + "/"))

    zip_files = [b.name for b in blobs if b.name.endswith(".zip")]
    if zip_files:
        print("Detected ZIP SavedModel:", zip_files[0])
        return zip_files[0]

    keras_files = [b.name for b in blobs if b.name.endswith(".keras")]
    if keras_files:
        print("Detected Vertex .keras model:", keras_files[0])
        return keras_files[0]

    h5_files = [b.name for b in blobs if b.name.endswith(".h5")]
    if h5_files:
        print("Detected H5 model:", h5_files[0])
        return h5_files[0]

    raise FileNotFoundError(f"Aucun fichier modèle trouvé dans {model_folder}")

# Chargement du modèle + scaler + dataset + config
def load_artifacts_gcs(bucket_name, base_path, run_id=None):
    model_folder = get_model_folder(bucket_name, base_path, run_id)

    model_blob = find_model_file(bucket_name, model_folder)
    model = load_model_from_gcs_safely(bucket_name, model_blob)

    scaler_blob = f"{model_folder}/scaler.pkl"
    parquet_blob = f"{model_folder}/train_data.parquet"
    config_blob = f"{model_folder}/model_config.json"

    scaler = joblib.load(io.BytesIO(load_from_gcs(bucket_name, scaler_blob)))
    df = pd.read_parquet(io.BytesIO(load_from_gcs(bucket_name, parquet_blob)))

    config_bytes = load_from_gcs(bucket_name, config_blob)
    config = json.loads(config_bytes.decode("utf-8"))

    return model, df, scaler, config, model_folder

# Construction des séquences LSTM
def create_sequences(df, lookback):
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--gcs_path", required=True)
    parser.add_argument("--run_id", required=False)
    args = parser.parse_args()

    print("\n=== Loading artifacts from GCS ===")
    model, df, scaler, config, model_folder = load_artifacts_gcs(
        args.bucket, args.gcs_path, args.run_id
    )

    # REPLACE DATAFRAME WITH RAW DATA
    print("Loading RAW data for this RUN_ID...")

    raw_blob = f"{args.gcs_path}/BTC_USDT_raw_{args.run_id}.parquet"
    raw_bytes = load_from_gcs(args.bucket, raw_blob)
    raw_df = pd.read_parquet(io.BytesIO(raw_bytes))

    df = raw_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.set_index("timestamp")

    cols = ["open", "high", "low", "close", "volume"]
    raw_vals = df[cols].values
    norm_vals = scaler.transform(raw_vals)

    df_norm = df.copy()
    df_norm[cols] = norm_vals

    print("RAW dataset loaded:", df.shape)

    lookback = config["lookback"]
    n_features = config["n_features"]
    print(f"Loaded model config: lookback={lookback}, n_features={n_features}")

    run_id = os.path.basename(model_folder)
    out_dir = os.path.join("evaluation", "plots_gcs", run_id)
    ensure_dir(out_dir)

    # Construction des séquences (normalisées)
    X, y = create_sequences(df_norm, lookback)

    raw = model(X)
    if isinstance(raw, dict):
        raw = raw["output_0"]

    preds = raw.numpy().reshape(-1)

    # Métriques normalisées
    mae = float(np.mean(np.abs(preds - y)))
    mse = float(np.mean((preds - y) ** 2))
    rmse = float(np.sqrt(mse))

    # Dénormalisation (CORRIGÉ)
    features_full_norm = norm_vals
    real_full_norm = features_full_norm[lookback:]
    pred_full_norm = real_full_norm.copy()
    pred_full_norm[:, 3] = preds

    real_prices = scaler.inverse_transform(real_full_norm)[:, 3]
    pred_prices = scaler.inverse_transform(pred_full_norm)[:, 3]
    dates = df.index[lookback:]

    # Prédiction t+1 (CORRIGÉ)
    last_seq = features_full_norm[-lookback:].reshape(1, lookback, n_features)

    raw_future = model(last_seq)
    if isinstance(raw_future, dict):
        raw_future = raw_future["output_0"]

    future_pred_norm = float(raw_future.numpy()[0][0])

    future_point = real_full_norm[-1:].copy()
    future_point[:, 3] = future_pred_norm
    future_usd = float(scaler.inverse_transform(future_point)[0, 3])

    delta_t = dates[-1] - dates[-2]
    future_time = dates[-1] + delta_t

    mae_usd = float(np.mean(np.abs(pred_prices - real_prices)))
    rmse_usd = float(np.sqrt(np.mean((pred_prices - real_prices) ** 2)))

    valid = real_prices != 0
    mape = float(np.mean(np.abs((real_prices[valid] - pred_prices[valid]) / real_prices[valid])) * 100)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "mae_usd": mae_usd,
            "rmse_usd": rmse_usd,
            "mape_usd": mape,
            "mae_normalized": mae,
            "mse_normalized": mse,
            "rmse_normalized": rmse
        }, f, indent=4)

    metric_text = (
        f"MAE (USD): {mae_usd:.4f}\n"
        f"RMSE (USD): {rmse_usd:.4f}\n"
        f"MAPE (USD): {mape:.2f}%\n\n"
        f"Last real: {real_prices[-1]:.2f} USD\n"
        f"Last pred: {pred_prices[-1]:.2f} USD\n"
        f"T+1 pred: {future_usd:.2f} USD"
    )

    plt.figure(figsize=(16, 6))
    plt.plot(dates, real_prices, label="Real Price (USD)")
    plt.plot(dates, pred_prices, label="Prediction (USD)")
    plt.scatter(future_time, future_usd, s=60, color="red", label="T+1 prediction")

    plt.gca().text(
        0.02, 0.98, metric_text,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.7)
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Predictions vs Real (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preds_usd.png"))
    plt.close()

    zoom = 200
    plt.figure(figsize=(16, 6))
    plt.plot(dates[-zoom:], real_prices[-zoom:], label="Real Price (USD)")
    plt.plot(dates[-zoom:], pred_prices[-zoom:], label="Prediction (USD)")
    plt.scatter(future_time, future_usd, s=60, color="red", label="T+1 prediction")

    plt.gca().text(
        0.02, 0.98, metric_text,
        transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.7)
    )

    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Zoom last 200 points")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preds_usd_zoom_200.png"))
    plt.close()

    with open(os.path.join(out_dir, "future_prediction.txt"), "w") as f:
        f.write(f"{future_time}, {future_usd}")

    print("\nEvaluation saved to:", out_dir)


if __name__ == "__main__":
    main()
