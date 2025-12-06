import os
import io
import time
import yaml
import pandas as pd
import ccxt
import joblib
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage

# Load project configuration from params.yaml
FILE_DIR = os.path.dirname(__file__)
PARAMS_PATH = os.path.join(FILE_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    config = yaml.safe_load(f)

# Extract feeder and training sections from the configuration
FEEDER = config["feeder"]
TRAINING = config["training"]

# Feeder parameters
PAIRS = FEEDER["pairs"]
INTERVAL = FEEDER["interval"]
DAYS = FEEDER.get("days", 90)

# GCS bucket configuration
BUCKET_URI = FEEDER["bucket"].rstrip("/")
BUCKET = BUCKET_URI.replace("gs://", "")
OUTPUT_PATH = FEEDER.get("output_path", "ohlcv-data")

# API and GCS clients
exchange = ccxt.kucoin()
storage_client = storage.Client()

def fetch_ohlcv(pair, since, timeframe):
    """Fetch OHLCV data from KuCoin since the given timestamp."""
    all_data = []

    while since < time.time() * 1000:
        print(f"Fetching {pair} since {datetime.fromtimestamp(since / 1000, tz=timezone.utc)}...")
        data = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=int(since))

        if not data:
            break

        all_data += data
        since = data[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

        if len(all_data) > 100000:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    print(f"{len(df)} candles fetched for {pair}.")
    return df

def normalize_data(df):
    """Normalize numeric columns using MinMaxScaler while keeping timestamp intact."""
    cols = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()

    df_norm = df.copy()
    df_norm[cols] = scaler.fit_transform(df[cols])
    return df_norm, scaler

def upload_to_gcs(df, path):
    """Upload a DataFrame to Google Cloud Storage in Parquet format."""
    bucket = storage_client.bucket(BUCKET)

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    blob = bucket.blob(path)
    blob.upload_from_file(buf, rewind=True)
    print(f"Uploaded to gs://{BUCKET}/{path}")

def main():
    # Compute start timestamp based on the number of historical days
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000

    # Training session identifier
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    print(f"\n=== RUN ID: {run_id} ===")

    for pair in PAIRS:
        print(f"\n=== Processing {pair} ===")

        # Raw historical OHLCV fetch
        df_raw = fetch_ohlcv(pair, since, INTERVAL)

        # Upload raw data
        raw_path = f"{OUTPUT_PATH}/{pair.replace('/', '_')}_raw_{run_id}.parquet"
        upload_to_gcs(df_raw, raw_path)

        # Normalize and upload train dataset
        df_train, scaler = normalize_data(df_raw)
        train_path = f"{OUTPUT_PATH}/{pair.replace('/', '_')}_train_{run_id}.parquet"
        upload_to_gcs(df_train, train_path)

        # Save and upload fitted scaler
        local_scaler_path = f"scaler_{pair.replace('/', '_')}_{run_id}.pkl"
        joblib.dump(scaler, local_scaler_path)

        bucket = storage_client.bucket(BUCKET)
        blob = bucket.blob(f"{OUTPUT_PATH}/scalers/{local_scaler_path}")
        blob.upload_from_filename(local_scaler_path)
        print(f"Scaler uploaded to gs://{BUCKET}/{OUTPUT_PATH}/scalers/{local_scaler_path}")

        os.remove(local_scaler_path)

    # Indicate that training will be done later via Makefile
    print("\nFetch and preprocessing completed locally.")
    print("Training will be launched with: make train")
    print(f"Use this RUN_ID for training: {run_id}\n")

    # Save run_id locally for the Makefile
    with open("last_run_id.txt", "w") as f:
        f.write(run_id)

if __name__ == "__main__":
    main()
