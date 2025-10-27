import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss and MAE"""
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


def get_pred_vs_true_plot(model: tf.keras.Model, ds_test: tf.data.Dataset) -> plt.Figure:
    """Plot predictions vs true values"""
    y_true = []
    y_pred = []
    for x_batch, y_batch in ds_test:
        y_true.append(y_batch.numpy())
        y_pred.append(model.predict(x_batch, verbose=0))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    fig = plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True Close")
    plt.plot(y_pred, label="Predicted Close")
    plt.xlabel("Time step")
    plt.ylabel("Close Price")
    plt.title("Predictions vs True Values")
    plt.legend()
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

    # Load model and training history
    model_path = model_folder / "model.keras"
    model = tf.keras.models.load_model(model_path)
    model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()

    # Evaluate
    loss, mae = model.evaluate(ds_test, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation MAE: {mae:.4f}")

    # Save metrics
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": float(loss), "val_mae": float(mae)}, f, indent=2)

    # Plot training history
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    # Plot predictions vs true
    fig = get_pred_vs_true_plot(model, ds_test)
    fig.savefig(plots_folder / "pred_vs_true.png")

    print(f"\n✅ Evaluation metrics and plots saved at {evaluation_folder.absolute()}")


if __name__ == "__main__":
    main()
