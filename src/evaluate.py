import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile
import tempfile
import os
import pandas as pd
import joblib

SAMPLE_SIZE = 5000  # Nombre max de points pour les graphiques


# ============================================================
# Dataset reconstruction LSTM
# ============================================================
def create_sequences(df, lookback):
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])  # close normalisé
    return np.array(X), np.array(y)


# ============================================================
# PLOTS
# ============================================================
def get_training_plot(model_history: dict):
    epochs = range(1, len(model_history["val_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, model_history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    return fig


def plot_training_narrative(model_history: dict):
    val_loss = model_history["val_loss"]
    epochs = range(1, len(val_loss) + 1)

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = float(np.min(val_loss))

    fig = plt.figure(figsize=(12, 6))
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")
    plt.axhline(best_val, color="green", linestyle="--", label=f"Best Val Loss = {best_val:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_pred_vs_true_plot(y_true, y_pred):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True Close (USD)", linewidth=1)
    plt.plot(y_pred, label="Predicted Close (USD)", linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_error_over_time_plot(y_true, y_pred):
    abs_error = np.abs(y_true - y_pred)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(abs_error, linewidth=1)
    plt.title("Absolute Error Over Time (USD)")
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_error_distribution_plot(y_true, y_pred):
    errors = (y_true - y_pred)
    fig = plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50)
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_correlation_plot(y_true, y_pred):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.4)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    plt.xlabel("True Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    return fig


# ============================================================
# Collect predictions (convert to USD)
# ============================================================
def collect_predictions(model, X, y, scaler):
    preds = model(X)
    if isinstance(preds, dict):
        preds = preds["output_0"]

    preds = preds.numpy().flatten()
    y = y.flatten()

    # Format 5 features for inverse_transform
    pred_full = np.zeros((len(preds), 5))
    true_full = np.zeros((len(y), 5))

    pred_full[:, 3] = preds
    true_full[:, 3] = y

    y_pred_usd = scaler.inverse_transform(pred_full)[:, 3]
    y_true_usd = scaler.inverse_transform(true_full)[:, 3]

    return y_true_usd[:SAMPLE_SIZE], y_pred_usd[:SAMPLE_SIZE]


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <model-folder> <prepared-dataset-folder>")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_folder = Path(sys.argv[2])

    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # 1) Load model_config.json (lookback)
    # -----------------------------------------
    config_file = model_folder / "model_config.json"
    if not config_file.exists():
        raise FileNotFoundError("model_config.json not found — cannot determine lookback.")

    with open(config_file, "r") as f:
        cfg = json.load(f)

    lookback = cfg["lookback"]
    print(f"[INFO] Model lookback detected = {lookback}")

    # -----------------------------------------
    # 2) Load RAW dataset
    # -----------------------------------------
    raw_path = prepared_folder / "raw.csv"
    if not raw_path.exists():
        raw_path = prepared_folder.parent / "raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError("raw.csv not found — prepare.py must produce raw.csv")

    df = pd.read_csv(raw_path)

    # -----------------------------------------
    # 3) Build sequences
    # -----------------------------------------
    X_test, y_test = create_sequences(df, lookback)

    # -----------------------------------------
    # 4) Load SavedModel (TFSMLayer)
    # -----------------------------------------
    saved_zip = next(model_folder.glob("saved_model_*.zip"))
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")

    with zipfile.ZipFile(saved_zip, "r") as z:
        z.extractall(tmpdir)

    model = tf.keras.layers.TFSMLayer(tmpdir, call_endpoint="serving_default")

    # -----------------------------------------
    # 5) Load scaler.pkl
    # -----------------------------------------
    scaler_path = model_folder / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.pkl missing in model folder — needed for USD evaluation.")

    scaler = joblib.load(scaler_path)

    # -----------------------------------------
    # 6) Predictions (in USD)
    # -----------------------------------------
    y_true, y_pred = collect_predictions(model, X_test, y_test, scaler)

    # -----------------------------------------
    # 7) Metrics (USD)
    # -----------------------------------------
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"mae_usd": mae, "mse_usd": mse}, f, indent=2)

    # -----------------------------------------
    # 8) Training plots
    # -----------------------------------------
    raw_history = np.load(model_folder / "history.npy", allow_pickle=True)
    model_history = {"val_loss": [h["val_loss"] for h in raw_history]}

    get_training_plot(model_history).savefig(plots_folder / "training_history.png")
    plot_training_narrative(model_history).savefig(plots_folder / "training_narrative.png")
    get_pred_vs_true_plot(y_true, y_pred).savefig(plots_folder / "pred_vs_true.png")
    get_error_over_time_plot(y_true, y_pred).savefig(plots_folder / "error_over_time.png")
    get_error_distribution_plot(y_true, y_pred).savefig(plots_folder / "error_distribution.png")
    get_correlation_plot(y_true, y_pred).savefig(plots_folder / "correlation.png")

    print("Evaluation finished. Saved at:", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
