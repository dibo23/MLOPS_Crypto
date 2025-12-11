import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile
import tempfile
import pandas as pd
import joblib

SAMPLE_SIZE = 5000


# ============================================================
# Dataset reconstruction LSTM
# ============================================================
def create_sequences(df, lookback):
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])
    return np.array(X), np.array(y)


# ============================================================
# PLOTS
# ============================================================
def plot_training_history(model_history):
    epochs = range(1, len(model_history) + 1)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title("Training History")
    plt.grid(True)
    return fig


def plot_training_narrative(model_history):
    epochs = range(1, len(model_history) + 1)
    best_epoch = int(np.argmin(model_history)) + 1
    best_val = float(np.min(model_history))

    fig = plt.figure(figsize=(12, 5))
    plt.plot(epochs, model_history, linewidth=2, label="Val Loss")
    plt.axvline(best_epoch, linestyle="--", color="red", label=f"Best epoch = {best_epoch}")
    plt.axhline(best_val, linestyle="--", color="green", label=f"Best val_loss = {best_val:.4f}")
    plt.legend()
    plt.grid(True)
    return fig


def plot_loss_improvement(model_history):
    improvements = []
    for i in range(1, len(model_history)):
        prev = model_history[i - 1]
        curr = model_history[i]
        imp = (prev - curr) / prev if prev != 0 else 0
        improvements.append(imp)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(range(2, len(model_history) + 1), improvements)
    plt.title("Val Loss Improvement (%) per Epoch")
    plt.grid(True)
    return fig


def plot_early_stopping_signal(model_history, patience):
    best = np.inf
    no_imp = []
    counter = 0

    for v in model_history:
        if v < best:
            best = v
            counter = 0
        else:
            counter += 1
        no_imp.append(counter)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(no_imp, label="Epochs without improvement")
    plt.axhline(patience, color="red", linestyle="--",
                label=f"EarlyStopping patience={patience}")
    plt.legend()
    plt.grid(True)
    return fig


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

    # --------------------------------------------------------
    # Load lookback
    # --------------------------------------------------------
    config_path = model_folder / "model_config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    lookback = cfg["lookback"]
    print(f"[INFO] Detected lookback = {lookback}")

    # --------------------------------------------------------
    # Load raw.csv
    # --------------------------------------------------------
    raw_path = prepared_folder / "raw.csv"
    if not raw_path.exists():
        raw_path = prepared_folder.parent / "raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError("raw.csv NOT FOUND")

    df = pd.read_csv(raw_path)

    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------
    X_test, y_test = create_sequences(df, lookback)

    # --------------------------------------------------------
    # Load SavedModel (Vertex)
    # --------------------------------------------------------
    saved_zip = next(model_folder.glob("saved_model_*.zip"))
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")
    with zipfile.ZipFile(saved_zip, "r") as z:
        z.extractall(tmpdir)

    model = tf.keras.layers.TFSMLayer(tmpdir, call_endpoint="serving_default")

    # --------------------------------------------------------
    # Load scaler.pkl for USD metrics
    # --------------------------------------------------------
    scaler_path = model_folder / "scaler.pkl"
    scaler = joblib.load(scaler_path)

    # --------------------------------------------------------
    # Predict in USD
    # --------------------------------------------------------
    preds = model(X_test)
    if isinstance(preds, dict):
        preds = preds["output_0"]
    preds = preds.numpy().flatten()
    y_test = y_test.flatten()

    f_pred = np.zeros((len(preds), 5))
    f_true = np.zeros((len(y_test), 5))
    f_pred[:, 3] = preds
    f_true[:, 3] = y_test

    y_pred_usd = scaler.inverse_transform(f_pred)[:, 3]
    y_true_usd = scaler.inverse_transform(f_true)[:, 3]

    y_pred_usd = y_pred_usd[:SAMPLE_SIZE]
    y_true_usd = y_true_usd[:SAMPLE_SIZE]

    # --------------------------------------------------------
    # Write metrics.json
    # --------------------------------------------------------
    mae = float(np.mean(np.abs(y_pred_usd - y_true_usd)))
    mse = float(np.mean((y_pred_usd - y_true_usd) ** 2))

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"mae_usd": mae, "mse_usd": mse}, f, indent=2)

    # --------------------------------------------------------
    # Load training history
    # --------------------------------------------------------
    raw_history = np.load(model_folder / "history.npy", allow_pickle=True)
    val_loss_history = [h["val_loss"] for h in raw_history]

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------
    plot_training_history(val_loss_history).savefig(plots_folder / "training_history.png")
    plot_training_narrative(val_loss_history).savefig(plots_folder / "training_narrative.png")
    plot_loss_improvement(val_loss_history).savefig(plots_folder / "loss_improvement.png")
    plot_early_stopping_signal(val_loss_history, patience=4).savefig(
        plots_folder / "early_stopping_signal.png"
    )

    print("Evaluation finished. Saved at:", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
