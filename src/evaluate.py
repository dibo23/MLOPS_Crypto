import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SAMPLE_SIZE = 5000  # Nombre maximum de points pour les graphiques


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot training & validation loss and MAE."""
    epochs = range(1, len(model_history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(epochs, model_history["loss"], label="Training loss")
    ax1.plot(epochs, model_history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if "mean_absolute_error" in model_history:
        ax2 = ax1.twinx()
        ax2.plot(epochs, model_history["mean_absolute_error"], label="Training MAE", color="orange")
        ax2.plot(epochs, model_history["val_mean_absolute_error"], label="Validation MAE", color="red")
        ax2.set_ylabel("MAE")
        ax2.legend(loc="upper right")

    plt.title("Training and validation metrics")
    plt.tight_layout()
    return fig


def get_pred_vs_true_plot(model: tf.keras.Model, ds_test: tf.data.Dataset, sample_size=SAMPLE_SIZE):
    """Plot predictions vs true values."""
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
    """Plot absolute error over time."""
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
    """Plot distribution (histogram) of prediction errors."""
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
    """Scatter plot of predicted vs true values with y=x line."""
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("True Close")
    plt.ylabel("Predicted Close")
    plt.title("Predicted vs True Values")
    plt.grid(True)
    plt.tight_layout()
    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Load model and history
    model = tf.keras.models.load_model(model_folder / "model.keras")
    model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()

    # Evaluate
    loss, mae = model.evaluate(ds_test, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # Save metrics
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": float(loss), "val_mae": float(mae)}, f, indent=2)

    # === PLOTS ===
    # 1. Training history
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    # 2. Predictions vs True
    fig, y_true, y_pred = get_pred_vs_true_plot(model, ds_test)
    fig.savefig(plots_folder / "pred_vs_true.png")

    # 3. Error over time
    fig = get_error_over_time_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_over_time.png")

    # 4. Error distribution
    fig = get_error_distribution_plot(y_true, y_pred)
    fig.savefig(plots_folder / "error_distribution.png")

    # 5. Correlation plot
    fig = get_correlation_plot(y_true, y_pred)
    fig.savefig(plots_folder / "correlation.png")

    print(f"\n✅ Evaluation metrics and plots saved at {evaluation_folder.absolute()}")


if __name__ == "__main__":
    main()
