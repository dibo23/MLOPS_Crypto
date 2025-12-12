import io
import os
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
from google.cloud import storage


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BUCKET_NAME = "bucket-crypto-data"
PAIR_SLUG = "BTC_USDT"

EVAL_PATH = f"monitoring/{PAIR_SLUG}_eval.parquet"
STATE_PATH = f"monitoring/{PAIR_SLUG}_drift_state.json"

# Drift detection parameters
WINDOW_N = 288            # Example: 288 candles of 5 minutes = 24h
MSE_THRESHOLD = 3         # Drift threshold (to be tuned)
COOLDOWN_MIN = 360        # Cooldown period in minutes (6 hours)

# GitHub Actions configuration (environment variables)
GITHUB_OWNER = os.environ["GITHUB_OWNER"]
GITHUB_REPO = os.environ["GITHUB_REPO"]
GITHUB_WORKFLOW_FILE = os.environ["GITHUB_WORKFLOW_FILE"]  # e.g. "retrain.yml"
GITHUB_REF = os.environ.get("GITHUB_REF", "main")
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]  # PAT or GitHub App token


# Optional inputs for the GitHub workflow
WORKFLOW_INPUTS = {
    "pair": PAIR_SLUG,
}


# ---------------------------------------------------------------------------
# GCS HELPERS
# ---------------------------------------------------------------------------

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


def load_parquet_gcs(path: str) -> pd.DataFrame:
    """
    Load a Parquet file from Google Cloud Storage into a pandas DataFrame.
    """
    buffer = io.BytesIO()
    bucket.blob(path).download_to_file(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)


def load_state() -> dict:
    """
    Load the drift state JSON from GCS.
    """
    blob = bucket.blob(STATE_PATH)
    if not blob.exists():
        return {}
    return json.loads(blob.download_as_text())


def save_state(state: dict) -> None:
    """
    Persist the drift state JSON to GCS.
    """
    bucket.blob(STATE_PATH).upload_from_string(
        json.dumps(state, indent=2),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# GITHUB WORKFLOW TRIGGER
# ---------------------------------------------------------------------------

def trigger_github_workflow(reason: str, mse_value: float) -> None:
    """
    Trigger a GitHub Actions workflow via the workflow_dispatch event.
    """
    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/"
        f"{GITHUB_WORKFLOW_FILE}/dispatches"
    )

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    payload = {
        "ref": GITHUB_REF,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code not in (201, 204):
        raise RuntimeError(
            f"GitHub workflow dispatch failed: "
            f"{response.status_code} {response.text}"
        )

    print("GitHub Actions workflow successfully triggered.")


# ---------------------------------------------------------------------------
# DRIFT CHECK
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main drift detection routine.
    """
    df = load_parquet_gcs(EVAL_PATH)

    if df.empty or "sq_error" not in df.columns:
        print("No evaluation data found or 'sq_error' column is missing.")
        return

    df = df.sort_values("timestamp")
    recent = df.tail(WINDOW_N)

    if recent.empty:
        print("Recent evaluation window is empty.")
        return

    mse_recent = float(recent["sq_error"].mean())
    print(f"Recent MSE (N={len(recent)}): {mse_recent:.6g}")

    state = load_state()

    last_trigger = state.get("last_trigger_utc")
    last_trigger_dt = None
    if last_trigger:
        last_trigger_dt = datetime.fromisoformat(
            last_trigger.replace("Z", "+00:00")
        )

    now = datetime.now(timezone.utc)

    in_cooldown = (
        last_trigger_dt is not None
        and (now - last_trigger_dt) < timedelta(minutes=COOLDOWN_MIN)
    )

    if mse_recent > MSE_THRESHOLD and not in_cooldown:
        reason = (
            f"drift detected: mse_recent={mse_recent} "
            f"> threshold={MSE_THRESHOLD}"
        )

        trigger_github_workflow(
            reason=reason,
            mse_value=mse_recent,
        )

        state["last_trigger_utc"] = now.isoformat().replace("+00:00", "Z")
        state["last_mse"] = mse_recent
        state["threshold"] = MSE_THRESHOLD
        save_state(state)

    else:
        print(
            "No retraining triggered "
            "(threshold not exceeded or cooldown active)."
        )

        state["last_mse"] = mse_recent
        state["threshold"] = MSE_THRESHOLD
        save_state(state)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GitHub owner:", GITHUB_OWNER)
    print("GitHub repository:", GITHUB_REPO)
    print("Workflow file:", GITHUB_WORKFLOW_FILE)
    print("Git reference:", GITHUB_REF)

    main()
