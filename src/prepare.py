import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from utils.seed import set_seed


def get_preview_plot(df: pd.DataFrame) -> plt.Figure:
    """Affiche un aperçu de la série temporelle (prix de clôture)"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if "close" in df.columns:
        ax.plot(df["timestamp"], df["close"], label="Prix de clôture (close)")
        ax.set_ylabel("Prix de clôture")
    else:
        # Si pas de colonne 'close', on trace simplement la première colonne numérique
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        ax.plot(df["timestamp"], df[num_cols[0]], label=num_cols[0])
        ax.set_ylabel(num_cols[0])

    ax.set_title("Aperçu des données financières")
    ax.set_xlabel("Date")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def load_dataset_from_csv(csv_path: Path, test_split: float, seed: int):
    """Charge les données OHLCV depuis un fichier CSV"""
    df = pd.read_csv(csv_path)

    # Conversion de la colonne timestamp en datetime si elle existe
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Détection automatique des colonnes numériques (exclut symbol/timeframe)
    exclude_cols = {"timestamp", "timeframe", "symbol"}
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
    ]

    if not numeric_cols:
        raise ValueError("Aucune colonne numérique détectée dans le CSV !")

    print(f"\n✅ Colonnes utilisées pour l'entraînement : {numeric_cols}")

    # Données features (X) et cible (y)
    X = df[numeric_cols].values.astype("float32")

    # Exemple : la cible est 'close' si elle existe, sinon la dernière colonne numérique
    target_col = "close" if "close" in df.columns else numeric_cols[-1]
    y = df[target_col].values.astype("float32")

    # Découpage train/test sans mélanger (important pour les séries temporelles)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, shuffle=False
    )

    # Conversion en TensorFlow Datasets
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

    return df, ds_train, ds_test, numeric_cols


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <dataset.csv> <prepared-dataset-folder>\n")
        exit(1)

    # Chargement des paramètres
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    csv_path = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    seed = prepare_params.get("seed", 42)
    split = prepare_params.get("split", 0.2)

    # Initialisation de la graine aléatoire
    set_seed(seed)

    # Lecture et préparation des données
    df, ds_train, ds_test, numeric_cols = load_dataset_from_csv(csv_path, split, seed)

    # Création du dossier de sortie
    prepared_dataset_folder.mkdir(parents=True, exist_ok=True)

    # Graphique d’aperçu
    preview_plot = get_preview_plot(df)
    preview_plot.savefig(prepared_dataset_folder / "preview.png")

    # Normalisation des données
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(df[numeric_cols].values)
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))

    # Sauvegarde des datasets
    tf.data.Dataset.save(ds_train, str(prepared_dataset_folder / "train"))
    tf.data.Dataset.save(ds_test, str(prepared_dataset_folder / "test"))

    # Sauvegarde des métadonnées
    metadata = {
        "features": numeric_cols,
        "split": split,
        "seed": seed,
        "n_samples": len(df),
    }
    with open(prepared_dataset_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Données préparées et sauvegardées dans {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()
