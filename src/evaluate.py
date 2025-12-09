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

SAMPLE_SIZE = 5000  # Nombre maximum de points pour les graphiques


def create_sequences(df, lookback):
    """Reconstruit les séquences LSTM à partir d’un DataFrame normalisé."""
    cols = ["open", "high", "low", "close", "volume"]
    data = df[cols].values.astype("float32")

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 3])  # close

    return np.array(X), np.array(y)


def get_training_plot(model_history: dict) -> plt.Figure:
    """Génère la figure des courbes loss / val_loss."""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, model_history["loss"], label="Training loss")
    ax1.plot(epochs, model_history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    plt.title("Training and validation metrics")
    plt.tight_layout()
    return fig


def plot_training_narrative(model_history: dict) -> plt.Figure:
    """Graphique montrant val_loss + best epoch."""
    loss = model_history["loss"]
    val_loss = model_history["val_loss"]
    epochs = range(1, len(loss) + 1)

    best_epoch = int(np.argmin(val_loss)) + 1
    best_val = float(np.min(val_loss))

    fig = plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)

    plt.axvline(best_epoch, color="red", linestyle="--", label=f"Best Epoch = {best_epoch}")
    plt.axhline(best_val, color="green", linestyle="--", label=f"Best Val Loss = {best_val:.4f}")

    plt.title("Training Narrative: Loss, Val Loss, Early Stopping, Best Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return fig


def get_pred_vs_true_plot(model, X_test, y_test, sample_size=SAMPLE_SIZE):
    """Génère y_true vs y_pred sur le test."""
    preds = model(X_test)["output_0"].numpy().flatten()

    y_true = y_test[:sample_size]
    y_pred = preds[:sample_size]

    fig = plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True Close", linewidth=1)
    plt.plot(y_pred, label="Predicted Close", linewidth=1)
    plt.xlabel("Time step")
    plt.ylabel("Close Price")
    plt.title("Predictions vs True Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return fig, y_true, y_pred


def get_error_over_time_plot(y_true, y_pred):
    abs_error = np.abs(y_true - y_pred)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(abs_error, linewidth=1, color="purple")
    plt.title("Absolute Error Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_error_distribution_plot(y_true, y_pred):
    errors = (y_true - y_pred)

    fig = plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="teal", alpha=0.7)
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    return fig


def get_correlation_plot(y_true, y_pred):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")

    plt.xlabel("True Close")
    plt.ylabel("Predicted Close")
    plt.title("Predicted vs True Values")
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_all_tested_configs(all_candidates):
    fig = plt.figure(figsize=(14, 7))
    for i, cand in enumerate(all_candidates):
        history = cand["history"]
        epochs = [h["epoch"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        label = f"{i + 1}: " + ", ".join(f"{k}={v}" for k, v in cand["config"].items())
        plt.plot(epochs, val_losses, marker="o", label=label)

    plt.title("Validation Loss for ALL Hyperparameter Configurations Tested")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig


def main():
    if len(sys.argv) != 3:
        print("Arguments error. Usage: python3 evaluate.py <model-folder> <test-folder>")
        exit(1)

    model_folder = Path(sys.argv[1])
    test_folder = Path(sys.argv[2])

    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # 1) Chargement du test parquet
    test_parquet = test_folder / "test.parquet"
    scaler_path = test_folder / "scaler.pkl"
    metadata_path = test_folder / "metadata.json"

    if not test_parquet.exists():
        raise FileNotFoundError(f"Missing test file: {test_parquet}")

    df_test = pd.read_parquet(test_parquet)

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    scaler = joblib.load(scaler_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    lookback = metadata["lookback"]

    # Normalisation identique au train
    cols = ["open", "high", "low", "close", "volume"]
    df_test[cols] = scaler.transform(df_test[cols])

    # Reconstruction des séquences test
    X_test, y_test = create_sequences(df_test, lookback)

    # 2) Affichage des hyperparamètres testés
    candidates_path = model_folder / "all_candidates.json"
    if candidates_path.exists():
        with open(candidates_path, "r") as f:
            all_candidates = json.load(f)
        fig = plot_all_tested_configs(all_candidates)
        fig.savefig(plots_folder / "all_hyperparams.png")

    # 3) Chargement SavedModel ZIP
    savedmodel_zip = next(model_folder.glob("saved_model_*.zip"))
    tmpdir = tempfile.mkdtemp(prefix="savedmodel_")
    with zipfile.ZipFile(savedmodel_zip, "r") as z:
        z.extractall(tmpdir)

    model = tf.keras.layers.TFSMLayer(tmpdir, call_endpoint="serving_default")

    # 4) Chargement de l'historique
    raw_history = np.load(model_folder / "history.npy", allow_pickle=True)
    model_history = {
        "loss": [0] * len(raw_history),
        "val_loss": [h["val_loss"] for h in raw_history]
    }

    # 5) Prédictions complètes
    fig, y_true, y_pred = get_pred_vs_true_plot(model, X_test, y_test)
    fig.savefig(plots_folder / "pred_vs_true.png")

    # 6) Évaluation manuelle
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"mae": mae, "mse": mse}, f, indent=2)

    # 7) Graphiques finaux
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    fig = plot_training_narrative(model_history)
    fig.savefig(plots_folder / "training_narrative.png")

    fig = get_error_over_time_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_over_time.png")

    fig = get_error_distribution_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_distribution.png")

    fig = get_correlation_plot(y_true, y_pred)
    fig.savefig(plots_folder / "correlation.png")

    print("Evaluation finished. Saved in:", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
