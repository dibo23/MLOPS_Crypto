import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SAMPLE_SIZE = 5000  # Nombre maximum de points pour les graphiques


def get_training_plot(model_history: dict) -> plt.Figure:
    """Génère la figure des courbes loss / val_loss / MAE."""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, model_history["loss"], label="Training loss")
    ax1.plot(epochs, model_history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if "mean_absolute_error" in model_history:
        # Axe pour le MAE
        ax2 = ax1.twinx()
        ax2.plot(epochs, model_history["mean_absolute_error"], label="Training MAE", color="orange")
        ax2.plot(epochs, model_history["val_mean_absolute_error"], label="Validation MAE", color="red")
        ax2.set_ylabel("MAE")
        ax2.legend(loc="upper right")

    plt.title("Training and validation metrics")
    plt.tight_layout()
    return fig


def get_pred_vs_true_plot(model: tf.keras.Model, ds_test: tf.data.Dataset, sample_size=SAMPLE_SIZE):
    """Génère un plot y_true vs y_pred (valeurs normalisées)."""
    y_true, y_pred = [], []

    for x_batch, y_batch in ds_test:
        if len(y_true) >= sample_size:
            break
        y_true.append(y_batch.numpy())
        y_pred.append(model.predict(x_batch, verbose=0))

    y_true = np.concatenate(y_true)[:sample_size].flatten()
    y_pred = np.concatenate(y_pred)[:sample_size].flatten()

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


def get_error_over_time_plot(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Génère le graphique de l'erreur absolue au fil du temps."""
    abs_error = np.abs(y_true - y_pred)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(abs_error, color="purple", linewidth=1)
    plt.title("Absolute Error Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.tight_layout()

    return fig


def get_error_distribution_plot(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Histogramme de la distribution des erreurs."""
    errors = (y_true - y_pred).flatten()

    fig = plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="teal", alpha=0.7)
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()

    return fig


def get_correlation_plot(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Scatter plot y_true vs y_pred avec une ligne y=x."""
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    plt.xlabel("True Close")
    plt.ylabel("Predicted Close")
    plt.title("Predicted vs True Values")
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_all_tested_configs(all_candidates):
    """Plot des val_loss pour toutes les configs testées."""
    fig = plt.figure(figsize=(14, 7))

    for i, cand in enumerate(all_candidates):
        history = cand["history"]
        epochs = [h["epoch"] for h in history]
        val_losses = [h["val_loss"] for h in history]

        label = f"{i + 1}: " + ", ".join(f"{k}={v}" for k, v in cand["config"].items())
        plt.plot(epochs, val_losses, marker='o', label=label)

    plt.title("Validation Loss for ALL Hyperparameter Configurations Tested")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:")
        print("python3 evaluate.py <model-folder> <prepared-dataset-folder>")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])

    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Chargement du dataset test
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Affichage des configs d’hyperparamètres testées
    candidates_path = model_folder / "all_candidates.json"
    if candidates_path.exists():
        with open(candidates_path, "r") as f:
            all_candidates = json.load(f)

        fig = plot_all_tested_configs(all_candidates)
        fig.savefig(plots_folder / "all_hyperparams.png")

    # Chargement du modèle
    model = tf.keras.models.load_model(model_folder / "model.keras")

    # Chargement de l'historique
    model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()

    # Évaluation brute
    loss, mae = model.evaluate(ds_test, verbose=0)

    print(f"Validation loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # Sauvegarde des métriques
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": float(loss), "val_mae": float(mae)}, f, indent=2)

    # Graphique 1 : courbes d'entraînement
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    # Graphique 2 : vrai vs prédit
    fig, y_true, y_pred = get_pred_vs_true_plot(model, ds_test)
    fig.savefig(plots_folder / "pred_vs_true.png")

    # Graphique 3 : erreur dans le temps
    fig = get_error_over_time_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_over_time.png")

    # Graphique 4 : distribution des erreurs
    fig = get_error_distribution_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_distribution.png")

    # Graphique 5 : corrélation
    fig = get_correlation_plot(y_true, y_pred)
    fig.savefig(plots_folder / "correlation.png")

    print("Evaluation metrics and plots saved at", evaluation_folder.absolute())


if __name__ == "__main__":
    main()
