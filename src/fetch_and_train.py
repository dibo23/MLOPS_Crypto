import os
import io
import time
import yaml
import argparse
import pandas as pd
import ccxt
import joblib
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage

# Chargement des paramètres YAML du projet
FILE_DIR = os.path.dirname(__file__)
PARAMS_PATH = os.path.join(FILE_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    config = yaml.safe_load(f)

# Paramètres fetch OHLCV
FEEDER = config["feeder"]
INTERVAL = FEEDER["interval"]
DAYS = FEEDER.get("days", 90)

# Configuration GCS
BUCKET_URI = FEEDER["bucket"].rstrip("/")
BUCKET = BUCKET_URI.replace("gs://", "").split("/")[0]  # Correction
OUTPUT_PATH = FEEDER.get("output_path", "ohlcv-data")

# Exchange + client GCS
exchange = ccxt.kucoin()
storage_client = storage.Client()

# Args depuis GitHub Actions pour aligner RUN_ID
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True)
parser.add_argument("--pair", type=str, required=True)
args = parser.parse_args()

RUN_ID = args.run_id
PAIR_RAW = args.pair                  # Forme API : BTC/USDT
PAIR = args.pair.replace("/", "_")    # Forme GCS : BTC_USDT

def fetch_ohlcv(pair, since, timeframe):
    """Télécharge OHLCV depuis KuCoin."""
    all_data = []

    while since < time.time() * 1000:
        print(f"Fetching {pair} since {datetime.fromtimestamp(since/1000, tz=timezone.utc)}...")
        data = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=int(since))

        if not data:
            break

        all_data += data
        since = data[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

        if len(all_data) > 200000:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"{len(df)} rows fetched for {pair}")
    return df

def normalize_data(df):
    """Normalisation MinMax."""
    cols = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()

    df2 = df.copy()
    df2[cols] = scaler.fit_transform(df[cols])

    return df2, scaler

def upload_to_gcs(df, path):
    """Upload DataFrame vers GCS."""
    bucket = storage_client.bucket(BUCKET)

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    blob = bucket.blob(path)
    blob.upload_from_file(buf, rewind=True)
    print(f"Uploaded: gs://{BUCKET}/{path}")

def main():
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000

    print(f"\n=== Processing {PAIR} with RUN_ID {RUN_ID} ===")

    # 1) Récupération RAW
    df_raw = fetch_ohlcv(PAIR_RAW, since, INTERVAL)

    if df_raw.empty:
        raise ValueError(f"No data fetched for {PAIR_RAW}. Cannot continue.")

    upload_to_gcs(df_raw, f"{OUTPUT_PATH}/{PAIR}_raw_{RUN_ID}.parquet")

    # 2) Normalisation
    df_train, scaler = normalize_data(df_raw)
    upload_to_gcs(df_train, f"{OUTPUT_PATH}/{PAIR}_train_{RUN_ID}.parquet")

    # 3) Upload scaler
    local_scaler = f"scaler_{PAIR}_{RUN_ID}.pkl"
    joblib.dump(scaler, local_scaler)

    bucket = storage_client.bucket(BUCKET)
    bucket.blob(f"{OUTPUT_PATH}/scalers/{local_scaler}").upload_from_filename(local_scaler)
    os.remove(local_scaler)

    print("\nDataset + scaler uploaded successfully.\n")

if __name__ == "__main__":
    main()
