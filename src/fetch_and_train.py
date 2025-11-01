import os
import io
import time
import yaml
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
from google.cloud import storage
# Load YAML configuration
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract configuration sections
FEEDER = config["feeder"]
TRAINING = config["training"]

PAIRS = FEEDER["pairs"]
INTERVAL = FEEDER["interval"]
MONTHS = FEEDER.get("months", 3)
BUCKET_URI = FEEDER["bucket"].rstrip("/")  # e.g. gs://bucket-crypto-data
BUCKET = BUCKET_URI.replace("gs://", "")
OUTPUT_PATH = FEEDER.get("output_path", "ohlcv-data")

exchange = ccxt.kucoin()
storage_client = storage.Client()

def fetch_ohlcv(pair, since, timeframe):
    """Fetch historical OHLCV data from Binance starting at a given timestamp."""
    all_data = []
    while since < time.time() * 1000:
        print(f"Fetching {pair} since {datetime.fromtimestamp(since / 1000, tz=timezone.utc)}...")
        data = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=int(since))
        if not data:
            break
        all_data += data
        since = data[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
        if len(all_data) > 100000:  # safety stop
            break
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    print(f"{len(df)} candles fetched for {pair}.")
    return df

def upload_to_gcs(df, pair):
    """Upload the OHLCV dataframe to a Google Cloud Storage bucket."""
    print(f"Uploading {pair} data to GCS...")
    bucket = storage_client.bucket(BUCKET)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    blob = bucket.blob(f"{OUTPUT_PATH}/{pair.replace('/', '_')}.parquet")
    blob.upload_from_file(buf, rewind=True)
    print(f"Uploaded {len(df)} rows to {BUCKET_URI}/{OUTPUT_PATH}/{pair.replace('/', '_')}.parquet")

# def upload_to_gcs(df, pair):
#     """Upload locally for testing instead of GCS."""
#     os.makedirs("local_data", exist_ok=True)
#     file_path = f"local_data/{pair.replace('/', '_')}.parquet"
#     df.to_parquet(file_path, index=False)
#     print(f"✅ Saved {len(df)} rows locally to {file_path}")

def train_job():
    """Run the local training script."""
    print("Starting training job...")
    os.system(
        f"python3 src/train_lstm.py "
        f"--epochs {TRAINING['epochs']} "
        f"--batch_size {TRAINING['batch_size']} "
        f"--data_prefix {TRAINING['data_prefix']}"
    )

def main():
    since = (datetime.now(timezone.utc) - timedelta(days=30 * MONTHS)).timestamp() * 1000
    for pair in PAIRS:
        df = fetch_ohlcv(pair, since, INTERVAL)
        upload_to_gcs(df, pair)
    train_job()

if __name__ == "__main__":
    main()
