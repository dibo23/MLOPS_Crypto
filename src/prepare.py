import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import yaml
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils.seed import set_seed

def get_preview_plot(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["close"], label="Prix de clôture")
    ax.set_ylabel("Prix de clôture")
    ax.set_xlabel("Date")
    ax.set_title("Aperçu des données financières")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_lstm_sequences(X, y, lookback):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - lookback):
        sequences_X.append(X[i:i+lookback])
        sequences_y.append(y[i+lookback])
    return (
        tf.convert_to_tensor(sequences_X, dtype=tf.float32),
        tf.convert_to_tensor(sequences_y, dtype=tf.float32)
    )


# NEW — charge automatiquement le lookback depuis local_model_<RUN_ID>/model_config.json
def load_lookback_from_model_dir(model_dir: str):
    config_path = Path(model_dir) / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if "lookback" not in config:
        raise ValueError("lookback missing from model_config.json")

    lookback = int(config["lookback"])
    print(f"[INFO] Auto-detected lookback={lookback} from model_config.json")
    return lookback
# END NEW


def load_dataset_from_csv(csv_path: Path, test_split: float, seed: int, lookback: int):
    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    X = df[numeric_cols].values.astype("float32")
    y = df["close"].values.astype("float32")

    # Normalisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_split, random_state=seed, shuffle=False
    )

    X_train, y_train = create_lstm_sequences(X_train, y_train, lookback)
    X_test, y_test = create_lstm_sequences(X_test, y_test, lookback)

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return df, ds_train, ds_test, numeric_cols, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, required=False)

    # NEW — permet d'indiquer où trouver le model_config.json
    parser.add_argument("--model_dir", type=str, required=False,
                        help="Path to local_model_<RUN_ID> to auto-load lookback")
    # END NEW

    parser.add_argument("csv_path")
    parser.add_argument("output_folder")
    args = parser.parse_args()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARAMS_PATH = os.path.join(ROOT_DIR, "params.yaml")
    prepare_params = yaml.safe_load(open(PARAMS_PATH))["prepare"]

    csv_path = Path(args.csv_path)
    prepared_dataset_folder = Path(args.output_folder)

    seed = prepare_params.get("seed", 42)
    split = prepare_params.get("split", 0.2)

    # NEW — priorité:
    # 1) --lookback CLI
    # 2) model_config.json si --model_dir
    # 3) params.yaml sinon
    if args.lookback is not None:
        lookback = args.lookback
    elif args.model_dir:
        lookback = load_lookback_from_model_dir(args.model_dir)
    else:
        lookback = prepare_params.get("lookback", 30)
    # END NEW

    set_seed(seed)

    df, ds_train, ds_test, numeric_cols, scaler = load_dataset_from_csv(
        csv_path, split, seed, lookback
    )

    prepared_dataset_folder.mkdir(parents=True, exist_ok=True)

    preview_plot = get_preview_plot(df)
    preview_plot.savefig(prepared_dataset_folder / "preview.png")

    tf.data.Dataset.save(ds_train, str(prepared_dataset_folder / "train"))
    tf.data.Dataset.save(ds_test, str(prepared_dataset_folder / "test"))

    import joblib
    joblib.dump(scaler, prepared_dataset_folder / "scaler.pkl")

    metadata = {
        "features": numeric_cols,
        "split": split,
        "seed": seed,
        "lookback": lookback,
        "n_samples": len(df),
    }
    with open(prepared_dataset_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDonnées préparées avec lookback={lookback} dans {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()