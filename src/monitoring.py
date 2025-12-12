import io
import os
import re
import json
import zipfile
import shutil
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from google.cloud import storage


# ============================================================
# SETTINGS
# ============================================================

BUCKET_NAME = "bucket-crypto-data"
OUTPUT_PATH = "ohlcv-data"          # Location where models are stored
MONITORING_PATH = "monitoring"      # Location where monitoring parquet files are stored

# Fallback timeframe if train_data.parquet is missing
FALLBACK_TIMEFRAME = "1h"

# Default lookback if metrics.json is incomplete
DEFAULT_LOOKBACK = 30

# Mapping used to infer timeframe from seconds
SECONDS_TO_TIMEFRAME = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    21600: "6h",
    28800: "8h",
    43200: "12h",
    86400: "1d",
}

# Default OHLCV column order when column names cannot be inferred
DEFAULT_OHLCV_ORDER = ["open", "high", "low", "close", "volume"]

DATE_REGEX = re.compile(r"\d{8}-\d{6}")
EPOCH10_REGEX = re.compile(r"(?<!\d)\d{10}(?!\d)")

exchange = ccxt.kucoin()
storage_client = storage.Client()


# ============================================================
# GCS helpers
# ============================================================

def blob_exists(path: str) -> bool:
    """Check if a blob exists in the configured GCS bucket."""
    return storage_client.bucket(BUCKET_NAME).blob(path).exists()


def load_parquet_if_exists(path: str, columns=None) -> pd.DataFrame:
    """Load a parquet file from GCS if it exists, otherwise return an empty DataFrame."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    if not blob.exists():
        return pd.DataFrame(columns=columns) if columns else pd.DataFrame()

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    return pd.read_parquet(buffer)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as a parquet file to GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    blob.upload_from_file(buffer, rewind=True)
    print(f"Parquet file updated at gs://{BUCKET_NAME}/{path}")


def download_parquet(path: str) -> pd.DataFrame:
    """Download a parquet file from GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)

    return pd.read_parquet(buffer)


def load_metrics(folder: str) -> dict:
    """Load metrics.json from a model folder if it exists."""
    metrics_path = f"{folder}/metrics.json"

    if not blob_exists(metrics_path):
        return {}

    text = storage_client.bucket(BUCKET_NAME).blob(metrics_path).download_as_text()
    return json.loads(text)


# ============================================================
# Discover pairs from model folders
# Folder format: ohlcv-data/models/BTC_USDT_.../
# ============================================================

def discover_pair_slugs() -> List[str]:
    """Discover available trading pair slugs from model folders."""
    prefix = f"{OUTPUT_PATH}/models/"
    slugs = set()

    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefix):
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue

        folder_name = parts[2]
        if "_" not in folder_name:
            continue

        tokens = folder_name.split("_")
        if len(tokens) >= 2:
            slugs.add(f"{tokens[0]}_{tokens[1]}")

    return sorted(slugs)


# ============================================================
# Identify the latest model folder based on timestamps
# Example: BTC_USDT_1765459578_20251211-140853
# ============================================================

def folder_datetime(folder_path: str) -> datetime:
    """Extract the most recent datetime from a model folder name."""
    name = folder_path.split("/")[-1]

    dates = DATE_REGEX.findall(name)
    if dates:
        return max(datetime.strptime(d, "%Y%m%d-%H%M%S") for d in dates)

    epochs = EPOCH10_REGEX.findall(name)
    if epochs:
        return datetime.fromtimestamp(
            max(int(e) for e in epochs),
            tz=timezone.utc,
        ).replace(tzinfo=None)

    raise RuntimeError(f"Cannot parse date from folder name: {folder_path}")


def get_latest_model_folder(pair_slug: str) -> str:
    """Return the most recent model folder for a given trading pair."""
    prefix = f"{OUTPUT_PATH}/models/{pair_slug}_"
    folders = set()

    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=prefix):
        parts = blob.name.split("/")
        if len(parts) >= 3:
            folders.add("/".join(parts[:3]))

    if not folders:
        raise RuntimeError(f"No model folder found for pair {pair_slug}")

    latest_folder = max(folders, key=folder_datetime)
    print(f"Latest model folder selected for {pair_slug}: {latest_folder}")

    return latest_folder


# ============================================================
# Locate SavedModel zip and scaler file
# ============================================================

def find_saved_model_zip(folder: str) -> str:
    """Locate the saved_model zip file inside a model folder."""
    candidates = []

    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=f"{folder}/"):
        name = blob.name.lower()
        if name.endswith(".zip") and "saved_model" in name:
            candidates.append(blob.name)

    if not candidates:
        raise RuntimeError(
            f"No saved_model zip found under gs://{BUCKET_NAME}/{folder}/"
        )

    candidates.sort()
    return candidates[-1]


def find_scaler_pkl(folder: str) -> str:
    """Locate the scaler pickle file inside a model folder."""
    candidate = f"{folder}/scaler.pkl"
    if blob_exists(candidate):
        return candidate

    for blob in storage_client.list_blobs(BUCKET_NAME, prefix=f"{folder}/"):
        if blob.name.endswith(".pkl"):
            return blob.name

    raise RuntimeError(
        f"No scaler pickle file found under gs://{BUCKET_NAME}/{folder}/"
    )


# ============================================================
# Load SavedModel from zip archive (TensorFlow 2.19 compatible)
# ============================================================

def load_infer_from_savedmodel_zip(zip_blob_path: str, cache_key: str):
    """Load a TensorFlow SavedModel from a zipped archive stored in GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)

    base_dir = os.path.abspath(
        os.path.join(".cache", "saved_models", cache_key)
    )
    zip_path = os.path.join(base_dir, "model.zip")
    extract_dir = os.path.join(base_dir, "extracted")

    os.makedirs(base_dir, exist_ok=True)

    bucket.blob(zip_blob_path).download_to_filename(zip_path)

    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    model = tf.saved_model.load(extract_dir)

    if "serving_default" not in model.signatures:
        raise RuntimeError(
            f"SavedModel has no 'serving_default' signature: {zip_blob_path}"
        )

    return model.signatures["serving_default"]


def infer_predict(infer_fn, X_input: np.ndarray) -> float:
    """Run inference on a single input window and return a scalar prediction."""
    X_tf = tf.constant(X_input, dtype=tf.float32)

    _, kw = infer_fn.structured_input_signature
    if len(kw) == 1:
        input_name = next(iter(kw.keys()))
        out = infer_fn(**{input_name: X_tf})
    else:
        out = infer_fn(X_tf)

    y_tensor = next(iter(out.values()))
    y_np = y_tensor.numpy()

    return float(np.reshape(y_np, (-1,))[0])


# ============================================================
# Infer timeframe from train_data.parquet
# ============================================================

def infer_timeframe_from_train_data(folder: str) -> str:
    """Infer the timeframe from train_data.parquet timestamps."""
    train_path = f"{folder}/train_data.parquet"

    if not blob_exists(train_path):
        print(
            f"{train_path} not found. "
            f"Falling back to timeframe={FALLBACK_TIMEFRAME}."
        )
        return FALLBACK_TIMEFRAME

    df = download_parquet(train_path)

    ts_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        print(
            "train_data.parquet has no recognizable time column. "
            f"Using fallback timeframe={FALLBACK_TIMEFRAME}."
        )
        return FALLBACK_TIMEFRAME

    ts = (
        pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        .dropna()
        .sort_values()
    )

    if len(ts) < 3:
        print(
            "Not enough timestamps in train_data.parquet. "
            f"Using fallback timeframe={FALLBACK_TIMEFRAME}."
        )
        return FALLBACK_TIMEFRAME

    sec = int(ts.diff().dropna().dt.total_seconds().mode().iloc[0])
    timeframe = SECONDS_TO_TIMEFRAME.get(sec)

    if not timeframe:
        print(
            f"Interval of {sec} seconds is not mapped. "
            f"Using fallback timeframe={FALLBACK_TIMEFRAME}."
        )
        return FALLBACK_TIMEFRAME

    return timeframe


# ============================================================
# OHLCV incremental fetch
# ============================================================

def fetch_ohlcv_since(pair: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Fetch OHLCV data incrementally from the exchange."""
    all_candles = []
    limit = 1500
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while True:
        if since_ms >= now_ms:
            break

        batch = exchange.fetch_ohlcv(
            pair,
            timeframe=timeframe,
            since=since_ms,
            limit=limit,
        )

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
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return df


def update_prices_parquet(
    pair: str,
    pair_slug: str,
    timeframe: str,
    lookback: int,
) -> pd.DataFrame:
    """Update and persist OHLCV prices parquet used for monitoring."""
    prices_path = f"{MONITORING_PATH}/{pair_slug}_prices.parquet"
    columns = ["timestamp", "open", "high", "low", "close", "volume"]

    df_old = load_parquet_if_exists(prices_path, columns=columns)

    if df_old.empty:
        df_old = pd.DataFrame(columns=columns)

    now = datetime.now(timezone.utc)
    required_points = max(lookback + 1, lookback * 3)

    sec = next(
        (s for s, tfm in SECONDS_TO_TIMEFRAME.items() if tfm == timeframe),
        3600,
    )

    if df_old.empty or df_old["timestamp"].isna().all():
        since_start = now - timedelta(seconds=required_points * sec)
        since_ms = int(since_start.timestamp() * 1000)
        print(
            f"No existing prices. Fetching approximately {required_points} "
            f"steps starting from {since_start.isoformat()} "
            f"(timeframe={timeframe})."
        )
    else:
        last_ts = pd.to_datetime(df_old["timestamp"], utc=True).max()
        since_ms = int(last_ts.timestamp() * 1000)
        print(
            f"Prices available until {last_ts}. "
            f"Fetching missing data up to now (timeframe={timeframe})."
        )

    df_new = fetch_ohlcv_since(pair, timeframe, since_ms)

    if df_new.empty and df_old.empty:
        return df_old

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = (
        df_all.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )

    save_parquet(df_all, prices_path)

    return df_all


# ============================================================
# Feature inference from scaler
# ============================================================

def infer_features_from_scaler(
    scaler,
    df_prices: pd.DataFrame,
) -> List[str]:
    """
    Infer feature names expected by the scaler.

    Priority:
    - scaler.feature_names_in_ if available
    - scaler.n_features_in_ mapped to DEFAULT_OHLCV_ORDER
    """
    if hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
        missing = [c for c in features if c not in df_prices.columns]
        if missing:
            raise RuntimeError(
                f"Scaler expects features {features} "
                f"but they are missing in prices parquet: {missing}"
            )
        return features

    if hasattr(scaler, "n_features_in_"):
        n_features = int(scaler.n_features_in_)
        if n_features <= len(DEFAULT_OHLCV_ORDER):
            features = DEFAULT_OHLCV_ORDER[:n_features]
            missing = [c for c in features if c not in df_prices.columns]
            if missing:
                raise RuntimeError(
                    f"Inferred features {features} are missing "
                    f"in prices parquet: {missing}"
                )
            return features

        raise RuntimeError(
            f"Scaler expects {n_features} features, "
            f"but only up to {len(DEFAULT_OHLCV_ORDER)} are supported."
        )

    raise RuntimeError(
        "Unable to infer features from scaler "
        "(missing feature_names_in_ and n_features_in_)."
    )


# ============================================================
# Evaluation: next-step prediction aligned on timestamp t
# Window [t-lookback, t-1] predicts t
# ============================================================

def inverse_target(
    scaler,
    y_pred_scaled: float,
    n_features: int,
    target_idx: int,
) -> float:
    """Inverse-transform a single scaled target value."""
    tmp = np.zeros((1, n_features), dtype=float)
    tmp[0, target_idx] = y_pred_scaled

    inv = scaler.inverse_transform(tmp)
    return float(inv[0, target_idx])


def append_eval_rows(
    pair: str,
    pair_slug: str,
    df_prices: pd.DataFrame,
    infer_fn,
    scaler,
    model_folder: str,
    run_id: str,
    trained_ts: str,
    features: List[str],
    target: str,
    lookback: int,
):
    """Append new evaluation rows aligned with available price data."""
    eval_path = f"{MONITORING_PATH}/{pair_slug}_eval.parquet"
    eval_columns = [
        "timestamp",
        "pair",
        "y_true",
        "y_pred",
        "error",
        "sq_error",
        "input_end_timestamp",
        "prediction_time",
        "model_folder",
        "run_id",
        "trained_ts",
    ]

    df_eval_old = load_parquet_if_exists(
        eval_path,
        columns=eval_columns,
    )
    last_eval_ts = (
        df_eval_old["timestamp"].max()
        if not df_eval_old.empty
        else None
    )

    if len(df_prices) < lookback + 1:
        print(
            f"Not enough price points for evaluation "
            f"(required={lookback + 1}, available={len(df_prices)})."
        )
        return

    if target not in df_prices.columns:
        raise RuntimeError(
            f"Target column '{target}' not found in prices parquet."
        )

    if target not in features:
        raise RuntimeError(
            f"Target '{target}' is not part of features={features}. "
            "Inverse transformation is not possible."
        )

    target_idx = features.index(target)
    n_features = len(features)

    if (
        hasattr(scaler, "n_features_in_")
        and int(scaler.n_features_in_) != n_features
    ):
        raise RuntimeError(
            f"Scaler expects {scaler.n_features_in_} features "
            f"but received {n_features}: {features}"
        )

    now_utc = datetime.now(timezone.utc)
    new_rows = []

    for i in range(lookback, len(df_prices)):
        ts_t = df_prices.iloc[i]["timestamp"]

        if last_eval_ts is not None and ts_t <= last_eval_ts:
            continue

        window = df_prices.iloc[i - lookback:i]
        input_end_ts = window["timestamp"].iloc[-1]

        X_raw = window[features].values.astype(float)
        X_scaled = scaler.transform(X_raw)
        X_input = np.expand_dims(X_scaled, axis=0)

        y_pred_scaled = infer_predict(infer_fn, X_input)
        y_pred = inverse_target(
            scaler,
            y_pred_scaled,
            n_features,
            target_idx,
        )

        y_true = float(df_prices.iloc[i][target])
        error = y_true - y_pred

        new_rows.append({
            "timestamp": ts_t,
            "pair": pair,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": float(error),
            "sq_error": float(error * error),
            "input_end_timestamp": input_end_ts,
            "prediction_time": now_utc,
            "model_folder": model_folder,
            "run_id": run_id,
            "trained_ts": trained_ts,
        })

    if not new_rows:
        print("No new evaluation rows to append.")
        return

    df_new = pd.DataFrame(new_rows)
    df_all = pd.concat([df_eval_old, df_new], ignore_index=True)
    df_all = (
        df_all.drop_duplicates(subset=["timestamp", "model_folder"])
        .sort_values("timestamp")
    )

    save_parquet(df_all, eval_path)
    print(f"Added {len(df_new)} evaluation rows for {pair_slug}.")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main monitoring entry point."""
    pair_slugs = discover_pair_slugs()
    if not pair_slugs:
        raise RuntimeError(
            f"No trading pairs found under "
            f"gs://{BUCKET_NAME}/{OUTPUT_PATH}/models/."
        )

    print(f"Detected trading pairs: {pair_slugs}")

    for pair_slug in pair_slugs:
        pair = pair_slug.replace("_", "/")

        folder = get_latest_model_folder(pair_slug)
        metrics = load_metrics(folder)

        run_id = str(metrics.get("run_id", ""))
        trained_ts = str(metrics.get("timestamp", ""))

        lookback = int(
            metrics.get("best_config", {}).get(
                "lookback",
                DEFAULT_LOOKBACK,
            )
        )
        timeframe = infer_timeframe_from_train_data(folder)

        print(
            f"\nProcessing {pair} using model folder {folder}\n"
            f"run_id={run_id}, trained_ts={trained_ts}, "
            f"lookback={lookback}, timeframe={timeframe}"
        )

        df_prices = update_prices_parquet(
            pair,
            pair_slug,
            timeframe,
            lookback,
        )
        if df_prices.empty:
            print("No price data available. Skipping.")
            continue

        scaler_path = find_scaler_pkl(folder)
        buffer = io.BytesIO()
        storage_client.bucket(BUCKET_NAME).blob(
            scaler_path
        ).download_to_file(buffer)
        buffer.seek(0)
        scaler = joblib.load(buffer)

        features = infer_features_from_scaler(scaler, df_prices)
        target = "close"
        print(
            f"Inferred features from scaler: {features} "
            f"(target={target})"
        )

        zip_path = find_saved_model_zip(folder)
        cache_key = folder.replace("/", "_")
        print(
            f"Using SavedModel zip at "
            f"gs://{BUCKET_NAME}/{zip_path}"
        )

        infer_fn = load_infer_from_savedmodel_zip(
            zip_path,
            cache_key=cache_key,
        )

        append_eval_rows(
            pair=pair,
            pair_slug=pair_slug,
            df_prices=df_prices,
            infer_fn=infer_fn,
            scaler=scaler,
            model_folder=folder,
            run_id=run_id,
            trained_ts=trained_ts,
            features=features,
            target=target,
            lookback=lookback,
        )

    print("Monitoring run completed.")


if __name__ == "__main__":
    main()
