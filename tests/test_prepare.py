import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.prepare import prepare_dataset


def test_prepare_output_shapes(tmp_path):
    """Test that prepare_dataset() creates correct outputs and scaler."""

    # Fake raw.csv
    df = pd.DataFrame({
        "timestamp": pd.date_range(start="2020-01-01", periods=200, freq="1min"),
        "open": np.random.rand(200),
        "high": np.random.rand(200),
        "low": np.random.rand(200),
        "close": np.random.rand(200),
        "volume": np.random.rand(200),
    })

    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    # 🔥 Appelle la vraie fonction utilisée par evaluate & training
    prepare_dataset(raw_csv=str(raw_path), output_folder=str(prepared_dir), lookback=30)

    # Vérifie que raw.csv a été copié
    assert (prepared_dir / "raw.csv").exists(), "raw.csv not copied"

    # Vérifie scaler.pkl
    scaler_path = prepared_dir / "scaler.pkl"
    assert scaler_path.exists(), "Scaler missing"

    # Vérifie que le scaler fonctionne
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(df[["open", "high", "low", "close", "volume"]].values)

    assert scaled.shape == (200, 5), "Scaler output has wrong shape"
