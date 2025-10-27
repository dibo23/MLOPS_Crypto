import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from utils.seed import set_seed

def get_model(input_shape: int, dense_size: int) -> tf.keras.Model:
    """Create a simple MLP for numerical features"""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(dense_size, activation="relu"),
            tf.keras.layers.Dense(dense_size // 2, activation="relu"),
            tf.keras.layers.Dense(1),  # prédiction continue (close)
        ]
    )
    return model

def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <prepared-dataset-folder> <model-folder>\n")
        exit(1)

    # Load parameters
    train_params = yaml.safe_load(open("params.yaml"))["train"]
    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])

    seed = train_params.get("seed", 42)
    lr = train_params.get("lr", 0.001)
    epochs = train_params.get("epochs", 10)
    dense_size = train_params.get("dense_size", 128)

    # Set seed for reproducibility
    set_seed(seed)

    # Load datasets
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Infer input dimension
    for x_batch, y_batch in ds_train.take(1):
        input_dim = x_batch.shape[1]

    # Define the model
    model = get_model(input_dim, dense_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )
    model.summary()

    # Train the model
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    # Save the model and history
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / "model.keras"
    model.save(model_path)
    np.save(model_folder / "history.npy", history.history)

    print(f"\n✅ Model saved at {model_folder.absolute()}")

if __name__ == "__main__":
    main()
