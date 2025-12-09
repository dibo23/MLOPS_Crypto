import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import zipfile
import tempfile
import os

SAMPLE_SIZE = 5000  # Nombre max de points pour les graphiques


def get_training_plot(model_history: dict) -> plt.Figure:
    epochs = range(1, len(model_history["val_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, model_history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    return fig


def plot_training_narrative(model_history: dict) -> plt.Figure:
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


def collect_predictions(model, ds_test):
    y_true = []
    y_pred = []

    for x_batch, y_batch in ds_test:
        preds = model(x_batch)["output_0"].numpy().flatten()
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds)

        if len(y_true) > SAMPLE_SIZE:
            break

    y_true = np.array(y_true[:SAMPLE_SIZE])
    y_pred = np.array(y_pred[:SAMPLE_SIZE])

    return y_true, y_pred


def get_pred_vs_true_plot(y_true, y_pred):
    fig = plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True Close", linewidth=1)
    plt.plot(y_pred, label="Predicted Close", linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_error_over_time_plot(y_true, y_pred):
    abs_error = np.abs(y_true - y_pred)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(abs_error, linewidth=1)
    plt.title("Absolute Error Over Time")
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
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_all_tested_configs(all_candidates):
    fig = plt.figure(figsize=(14, 7))
    for i, cand in enumerate(all_candidates):
        epochs = [h["epoch"] for h in cand["history"]]
        val_losses = [h["val_loss"] for h in cand["history"]]
        plt.plot(epochs, val_losses, marker="o",
                 label=f"{i+1}: " + ", ".join(f"{k}={v}" for k,v in cand["config"].items()))
    plt.legend(fontsize=7)
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <model-folder> <prepared-dataset-folder>")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_folder = Path(sys.argv[2])

    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # 1) Charger dataset TFRecord
    ds_test = tf.data.Dataset.load(str(prepared_folder / "test"))

    # 2) Hyperparams testés (visualisation)
    candidates_file = model_folder / "all_candidates.json"
    if candidates_file.exists():
        with open(candidates_file, "r") as f:
            all_candidates = json.load(f)
        fig = plot_all_tested_configs(all_candidates)
        fig.savefig(plots_folder / "all_hyperparams.png")

    # 3) Charger SavedModel exporté
    saved_zip = next(model_folder.glob("saved_model_*.zip"))
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")

    with zipfile.ZipFile(saved_zip, "r") as z:
        z.extractall(tmpdir)

    model = tf.keras.layers.TFSMLayer(tmpdir, call_endpoint="serving_default")

    # 4) Charger l’historique HPO
    raw_history = np.load(model_folder / "history.npy", allow_pickle=True)
    model_history = {"val_loss": [h["val_loss"] for h in raw_history]}

    # 5) Predictions complètes
    y_true, y_pred = collect_predictions(model, ds_test)

    # 6) Calcul metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred)**2))

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"mae": mae, "mse": mse}, f, indent=2)

    # 7) Sauvegarde des plots
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    fig = plot_training_narrative(model_history)
    fig.savefig(plots_folder / "training_narrative.png")

    fig = get_pred_vs_true_plot(y_true, y_pred)
    fig.savefig(plots_folder / "pred_vs_true.png")

    fig = get_error_over_time_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_over_time.png")

    fig = get_error_distribution_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_distribution.png")

    fig = get_correlation_plot(y_true, y_pred)
    fig.savefig(plots_folder / "correlation.png")

    print("Evaluation finished. Saved at:", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
