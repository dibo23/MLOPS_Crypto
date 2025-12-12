import io
from datetime import datetime, timedelta, timezone

import ccxt
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from google.cloud import storage

# === Load YAML configuration ===
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

FEEDER = config["feeder"]
TRAINING = config["training"]

PAIRS = FEEDER["pairs"]
INTERVAL = FEEDER["interval"]
BUCKET_URI = FEEDER["bucket"].rstrip("/")  # ex: gs://bucket-crypto-data
BUCKET_NAME = BUCKET_URI.replace("gs://", "")
OUTPUT_PATH = FEEDER.get("output_path", "ohlcv-data")  # là où sont tes modèles
MONITORING_PATH = "monitoring"                   # OHLCV de monitoring
PREDICTIONS_PATH = f"{MONITORING_PATH}/predictions"

FEATURES = TRAINING.get("features", ["close"])
TARGET = TRAINING.get("target", FEATURES[0])
LOOKBACK = int(TRAINING.get("lookback", 30))
HORIZON_HOURS = float(TRAINING.get("horizon", 1))  # "3" -> 3.0
HORIZON_DELTA = timedelta(hours=HORIZON_HOURS)

exchange = ccxt.kucoin()
storage_client = storage.Client()

# Interval -> secondes (pour calculer une fenêtre initiale raisonnable)
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


# ---------------------------------------------------------------------------
# Utils GCS / Parquet
# ---------------------------------------------------------------------------

def load_parquet(path: str, columns=None) -> pd.DataFrame:
    """Charge un Parquet dans GCS, ou DataFrame vide s'il n'existe pas."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pd.read_parquet(buf)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame en Parquet dans GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    blob.upload_from_file(buf, rewind=True)

    print(f"✅ Data updated at gs://{BUCKET_NAME}/{path}")


# ---------------------------------------------------------------------------
# Fetch OHLCV incrémental
# ---------------------------------------------------------------------------

def fetch_ohlcv_since(pair: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """
    Récupère les bougies OHLCV à partir de 'since_ms' (ms epoch)
    en paginant jusqu'à maintenant.
    """
    all_candles = []
    limit = 1500  # limite KuCoin typique
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while True:
        if since_ms >= now_ms:
            break

        batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break

        all_candles.extend(batch)

        last_ts = batch[-1][0]
        if last_ts <= since_ms:
            break

        since_ms = last_ts + 1

        if last_ts >= now_ms:
            break

    if not all_candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def update_monitoring_ohlcv(pair: str) -> pd.DataFrame:
    """
    Met à jour le parquet monitoring/PAIR.parquet avec les nouvelles bougies
    et retourne le DataFrame complet.
    """
    now = datetime.now(timezone.utc)
    interval_seconds = INTERVAL_SECONDS.get(INTERVAL, 3600)

    pair_slug = pair.replace("/", "_")
    gcs_path = f"{MONITORING_PATH}/{pair_slug}.parquet"

    print(f"\n=== Update monitoring OHLCV pour {pair} (interval={INTERVAL}) ===")

    df_old = load_parquet(
        gcs_path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    if df_old.empty:
        # 1er run : juste assez de données pour quelques fenêtres (LOOKBACK)
        window_seconds = LOOKBACK * interval_seconds
        since_start = now - timedelta(seconds=window_seconds)
        since_ms = int(since_start.timestamp() * 1000)
        print(
            f"Aucun historique trouvé, fetch sur ~{LOOKBACK} pas en arrière "
            f"depuis {since_start.isoformat()}"
        )
    else:
        last_ts = df_old["timestamp"].max()
        since_ms = int(last_ts.timestamp() * 1000)
        print(f"Historique trouvé jusqu'à {last_ts}, fetch des données manquantes jusqu'à maintenant")

    df_new = fetch_ohlcv_since(pair, INTERVAL, since_ms)

    if df_new.empty and df_old.empty:
        print("Aucune donnée à enregistrer.")
        return df_old

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    save_parquet(df_all, gcs_path)
    return df_all


# ---------------------------------------------------------------------------
# Sélection du dernier modèle par timestamp de dossier
# ---------------------------------------------------------------------------

def get_latest_model_folder(pair_slug: str) -> str:
    """
    Retourne le chemin du dossier modèle le plus récent pour ce pair,
    en se basant sur le timestamp dans le nom de dossier.

    Dossiers de la forme:
      ohlcv-data/models/BTC_USDT_20251203-101344_20251202-231903/...
    """
    bucket = storage_client.bucket(BUCKET_NAME)
    prefix = f"{OUTPUT_PATH}/models/{pair_slug}_"

    folders = set()
    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefix):
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        folder = "/".join(parts[:3])   # "ohlcv-data/models/<foldername>"
        folders.add(folder)

    if not folders:
        raise RuntimeError(f"Aucun dossier modèle trouvé pour {pair_slug}")

    def extract_ts(folder_path: str) -> datetime:
        folder_name = folder_path.split("/")[-1]      # BTC_USDT_20251203-101344_20251202-231903
        # enlever "BTC_USDT_" au début
        # pair_slug = "BTC_USDT"
        rest = folder_name[len(pair_slug) + 1:]       # 20251203-101344_20251202-231903
        ts1 = rest.split("_")[0]                      # 20251203-101344
        return datetime.strptime(ts1, "%Y%m%d-%H%M%S")

    latest_folder = max(folders, key=extract_ts)
    print(f"📦 Dernier modèle détecté pour {pair_slug}: {latest_folder}")
    return latest_folder


def load_model_and_scaler(pair_slug: str):
    """Charge le dernier modèle + scaler pour le pair."""
    bucket = storage_client.bucket(BUCKET_NAME)
    folder = get_latest_model_folder(pair_slug)

    # modèle
    model_blob = bucket.blob(f"{folder}/lstm_model_{pair_slug}.keras")
    local_model = f"/tmp/lstm_model_{pair_slug}.keras"
    model_blob.download_to_filename(local_model)
    model = tf.keras.models.load_model(local_model)

    # scaler
    scaler_blob = bucket.blob(f"{folder}/scaler.pkl")
    buf = io.BytesIO()
    scaler_blob.download_to_file(buf)
    buf.seek(0)
    scaler = joblib.load(buf)

    return model, scaler, folder


# ---------------------------------------------------------------------------
# Prédictions multi-steps (rattrapage) avec horizon
# ---------------------------------------------------------------------------

def make_predictions_for_pair(df_all: pd.DataFrame, pair: str):
    """
    Fait toutes les prédictions manquantes pour ce pair, avec horizon en heures
    (training.horizon), et les append dans monitoring/predictions/PAIR_predictions.parquet.

    Si le job tourne toutes les 30min et interval=5m, il rattrape bien les ~6
    prédictions manquantes.
    """
    pair_slug = pair.replace("/", "_")

    if len(df_all) < LOOKBACK:
        print(f"Pas assez de données pour lookback={LOOKBACK}, skip prédictions.")
        return

    model, scaler, model_folder = load_model_and_scaler(pair_slug)

    pred_path = f"{PREDICTIONS_PATH}/{pair_slug}_predictions.parquet"
    pred_cols = [
        "prediction_time",
        "pair",
        "input_end_timestamp",
        "target_timestamp",
        "y_pred",
        "model_folder",
    ]
    df_pred_old = load_parquet(pred_path, columns=pred_cols)

    if df_pred_old.empty:
        last_target_ts = None
        # 1er run : on ne prédit que pour la dernière fenêtre
        indices = [len(df_all) - 1]
        print("Aucune prédiction existante, on crée une première prédiction.")
    else:
        last_target_ts = df_pred_old["target_timestamp"].max()
        # Rattrapage : toutes les fenêtres dont la target est > dernière prédite
        indices = range(LOOKBACK - 1, len(df_all))
        print(f"Dernière target prédite: {last_target_ts}, rattrapage des nouvelles.")

    new_rows = []
    now_utc = datetime.now(timezone.utc)

    for i in indices:
        window = df_all.iloc[i - LOOKBACK + 1 : i + 1]
        input_end_ts = window["timestamp"].iloc[-1]
        target_ts = input_end_ts + HORIZON_DELTA

        if last_target_ts is not None and target_ts <= last_target_ts:
            continue  # déjà prédit

        X_raw = window[FEATURES].values.astype(float)
        X_scaled = scaler.transform(X_raw)
        X_input = np.expand_dims(X_scaled, axis=0)  # (1, lookback, n_features)

        y_pred_scaled = model.predict(X_input, verbose=0).flatten()[0]

        if len(FEATURES) == 1 and FEATURES[0] == TARGET:
            inv = scaler.inverse_transform([[y_pred_scaled]])
            y_pred_real = float(inv[0, 0])
        else:
            y_pred_real = float(y_pred_scaled)

        row = {
            "prediction_time": now_utc,
            "pair": pair,
            "input_end_timestamp": input_end_ts,
            "target_timestamp": target_ts,
            "y_pred": y_pred_real,
            "model_folder": model_folder,
        }
        new_rows.append(row)

    if not new_rows:
        print("Aucune nouvelle prédiction à ajouter.")
        return

    df_new = pd.DataFrame(new_rows)
    df_pred_all = pd.concat([df_pred_old, df_new], ignore_index=True)
    df_pred_all = df_pred_all.drop_duplicates(subset=["target_timestamp"]).sort_values("target_timestamp")

    save_parquet(df_pred_all, pred_path)

    print(f"🔮 Ajouté {len(new_rows)} nouvelles prédictions pour {pair_slug} (horizon={HORIZON_HOURS}h).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    for pair in PAIRS:
        # 1) update OHLCV monitoring
        df_all = update_monitoring_ohlcv(pair)

        # 2) prédictions (rattrapage) avec le dernier modèle
        try:
            make_predictions_for_pair(df_all, pair)
        except Exception as e:
            print(f"⚠️ Impossible de faire les prédictions pour {pair}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
