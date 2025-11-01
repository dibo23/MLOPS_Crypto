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

# === Load YAML configuration ===
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Extract configuration sections ===
FEEDER = config["feeder"]
TRAINING = config["training"]

PAIRS = FEEDER["pairs"]
INTERVAL = FEEDER["interval"]
DAYS = FEEDER.get("days", 90)
BUCKET_URI = FEEDER["bucket"].rstrip("/")  # e.g. gs://bucket-crypto-data
BUCKET = BUCKET_URI.replace("gs://", "")
OUTPUT_PATH = FEEDER.get("output_path", "ohlcv-data")

exchange = ccxt.kucoin()
storage_client = storage.Client()

def fetch_ohlcv(pair, since, timeframe):
    """Fetch historical OHLCV data from KuCoin starting at a given timestamp."""
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
    print(f"✅ {len(df)} candles fetched for {pair}.")
    return df

def normalize_data(df):
    """Apply MinMax normalization to numeric columns."""
    cols = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[cols] = scaler.fit_transform(df[cols])
    return df_norm, scaler

def upload_to_gcs(df, path):
    """Upload a DataFrame as Parquet to GCS."""
    bucket = storage_client.bucket(BUCKET)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    blob = bucket.blob(path)
    blob.upload_from_file(buf, rewind=True)
    print(f"Uploaded to gs://{BUCKET}/{path}")

def launch_vertex_job(run_id, pair):
    job_name = f"lstm-train-{pair.replace('/', '_')}-{run_id}"
    region = "us-central1"
    bucket_code_path = f"gs://bucket-crypto-data/code/MLOPS_Crypto_{run_id}.tar.gz"

    # 1️⃣ Créer l’archive depuis ton code local
    os.system("tar -czf src.tar.gz src params.yaml setup.py")

    # 2️⃣ L’uploader sur ton bucket
    os.system(f"gsutil cp src.tar.gz {bucket_code_path}")

    # 3️⃣ Lancer le job Vertex AI
    cmd = (
        f"gcloud ai custom-jobs create "
        f"--region={region} "
        f"--display-name={job_name} "
        f"--python-package-uris={bucket_code_path} "
        f"--worker-pool-spec="
        f"machine-type=e2-standard-4,"
        f"executor-image-uri=us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest,"
        f"python-module=src.train_lstm "
        f"--args="
        f"--data_prefix={FEEDER['bucket']}/{FEEDER['output_path']},"
        f"--pair={pair.replace('/', '_')},"
        f"--run_id={run_id},"
        f"--epochs={TRAINING['epochs']},"
        f"--batch_size={TRAINING['batch_size']},"
        f"--lookback={TRAINING['lookback']}"
    )

    print(f"🚀 Launching Vertex AI job:\n{cmd}\n")
    os.system(cmd)

def main():
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    for pair in PAIRS:
        print(f"\n=== Processing {pair} ===")
        df_raw = fetch_ohlcv(pair, since, INTERVAL)

        # Save raw data
        raw_path = f"{OUTPUT_PATH}/{pair.replace('/', '_')}_raw_{run_id}.parquet"
        upload_to_gcs(df_raw, raw_path)

        # Normalize + save train data
        df_train, scaler = normalize_data(df_raw)
        train_path = f"{OUTPUT_PATH}/{pair.replace('/', '_')}_train_{run_id}.parquet"
        upload_to_gcs(df_train, train_path)

        # Save scaler
        local_scaler_path = f"scaler_{pair.replace('/', '_')}_{run_id}.pkl"
        joblib.dump(scaler, local_scaler_path)

        bucket = storage_client.bucket(BUCKET)
        blob = bucket.blob(f"{OUTPUT_PATH}/scalers/{local_scaler_path}")
        blob.upload_from_filename(local_scaler_path)
        print(f"✅ Scaler uploaded to gs://{BUCKET}/{OUTPUT_PATH}/scalers/{local_scaler_path}")
        os.remove(local_scaler_path)        

    launch_vertex_job(run_id, pair)
    #os.remove("src.tar.gz")

if __name__ == "__main__":
    main()
