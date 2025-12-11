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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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
# Direction-based metrics
# ============================================================
def compute_direction(arr):
    """Convert prices to direction: +1 = up, -1 = down."""
    diff = np.diff(arr)
    return np.where(diff >= 0, 1, -1)


def compute_direction_metrics(y_true_usd, y_pred_usd):
    t = compute_direction(y_true_usd)
    p = compute_direction(y_pred_usd)

    return {
        "direction_accuracy": float(accuracy_score(t, p)),
        "precision_up": float(precision_score(t, p, pos_label=1)),
        "precision_down": float(precision_score(t, p, pos_label=-1)),
        "recall_up": float(recall_score(t, p, pos_label=1)),
        "recall_down": float(recall_score(t, p, pos_label=-1)),
        "f1_up": float(f1_score(t, p, pos_label=1)),
        "f1_down": float(f1_score(t, p, pos_label=-1)),
    }


def plot_confusion_matrix(y_true_usd, y_pred_usd):
    t = compute_direction(y_true_usd)
    p = compute_direction(y_pred_usd)

    cm = confusion_matrix(t, p, labels=[-1, 1])
    cm_percent = cm / cm.sum() * 100

    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm_percent, cmap="Blues")
    plt.colorbar(label="Percentage (%)")

    plt.xticks([0, 1], ["DOWN", "UP"])
    plt.yticks([0, 1], ["DOWN", "UP"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha="center", va="center", color="black")

    plt.title("Confusion Matrix (%)")
    plt.tight_layout()
    return fig


# ============================================================
# Training curve plots
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

    # --------------------------------------------------------
    # Load raw dataset
    # --------------------------------------------------------
    raw_path = prepared_folder / "raw.csv"
    if not raw_path.exists():
        raw_path = prepared_folder.parent / "raw.csv"
    df = pd.read_csv(raw_path)

    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------
    X_test, y_test = create_sequences(df, lookback)

    # --------------------------------------------------------
    # Load SavedModel
    # --------------------------------------------------------
    saved_zip = next(model_folder.glob("saved_model_*.zip"))
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")
    with zipfile.ZipFile(saved_zip, "r") as z:
        z.extractall(tmpdir)
    model = tf.keras.layers.TFSMLayer(tmpdir, call_endpoint="serving_default")

    # --------------------------------------------------------
    # Load scaler.pkl
    # --------------------------------------------------------
    scaler = joblib.load(model_folder / "scaler.pkl")

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

    y_pred_usd = scaler.inverse_transform(f_pred)[:, 3][:SAMPLE_SIZE]
    y_true_usd = scaler.inverse_transform(f_true)[:, 3][:SAMPLE_SIZE]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    mae = float(np.mean(np.abs(y_pred_usd - y_true_usd)))
    mse = float(np.mean((y_pred_usd - y_true_usd) ** 2))

    direction = compute_direction_metrics(y_true_usd, y_pred_usd)

    all_metrics = {
        "mae_usd": mae,
        "mse_usd": mse,
        **direction
    }

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------
    raw_history = np.load(model_folder / "history.npy", allow_pickle=True)
    val_loss_history = [h["val_loss"] for h in raw_history]

    plot_training_history(val_loss_history).savefig(plots_folder / "training_history.png")
    plot_training_narrative(val_loss_history).savefig(plots_folder / "training_narrative.png")
    plot_loss_improvement(val_loss_history).savefig(plots_folder / "loss_improvement.png")
    plot_early_stopping_signal(val_loss_history, patience=4).savefig(
        plots_folder / "early_stopping_signal.png"
    )

    # Confusion Matrix
    plot_confusion_matrix(y_true_usd, y_pred_usd).savefig(
        plots_folder / "confusion_matrix.png"
    )

    # --------------------------------------------------------
    # HYPERPARAMETER ANALYSIS (if all_candidates.json exists)
    # --------------------------------------------------------
    hpo_file = model_folder / "all_candidates.json"
    hpo_plots_folder = plots_folder / "hpo"
    hpo_plots_folder.mkdir(parents=True, exist_ok=True)

    if hpo_file.exists():
        print("[INFO] Detected all_candidates.json → generating HPO plots")

        with open(hpo_file, "r") as f:
            candidates = json.load(f)

        maes = [c["mae"] for c in candidates]
        lookbacks = [c["lookback"] for c in candidates]
        lstm_units = [c["lstm_units"] for c in candidates]
        lrs = [c["learning_rate"] for c in candidates]

        # 1) BARPLOT — MAE per config
        fig = plt.figure(figsize=(10, 5))
        plt.bar(range(len(maes)), maes)
        plt.title("MAE per Hyperparameter Configuration")
        plt.xlabel("Configuration Index")
        plt.ylabel("MAE")
        plt.grid(True)
        fig.savefig(hpo_plots_folder / "barplot_mae.png")

        # 2) HEATMAP (lookback × lstm_units)
        unique_lb = sorted(list(set(lookbacks)))
        unique_lu = sorted(list(set(lstm_units)))
        heat = np.zeros((len(unique_lb), len(unique_lu)))

        for i, lb in enumerate(unique_lb):
            for j, lu in enumerate(unique_lu):
                vals = [
                    maes[k]
                    for k in range(len(maes))
                    if lookbacks[k] == lb and lstm_units[k] == lu
                ]
                heat[i, j] = np.mean(vals) if vals else np.nan

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(heat, cmap="viridis")
        plt.colorbar(label="Average MAE")
        plt.xticks(range(len(unique_lu)), unique_lu)
        plt.yticks(range(len(unique_lb)), unique_lb)
        plt.xlabel("LSTM units")
        plt.ylabel("Lookback")
        plt.title("HPO Heatmap (MAE)")
        plt.tight_layout()
        fig.savefig(hpo_plots_folder / "heatmap_mae.png")

        # 3) SCATTER — LR vs MAE
        fig = plt.figure(figsize=(8, 5))
        plt.scatter(lrs, maes)
        plt.xlabel("Learning Rate")
        plt.ylabel("MAE")
        plt.title("Learning Rate vs MAE")
        plt.grid(True)
        fig.savefig(hpo_plots_folder / "scatter_lr_mae.png")

        print("[INFO] HPO plots saved in:", hpo_plots_folder)

    print("Evaluation finished. Saved at:", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
