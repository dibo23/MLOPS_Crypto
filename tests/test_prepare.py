import pandas as pd
import numpy as np
from pathlib import Path
from src.prepare import main as prepare_main
import joblib

def test_prepare_output_shapes(tmp_path):

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

    # Appelle prepare.py via main()
    prepare_main([str(raw_path), str(prepared_dir)])

    # Vérifie outputs
    assert (prepared_dir / "raw.csv").exists(), "raw.csv not copied"
    assert (prepared_dir / "scaler.pkl").exists(), "Scaler missing"

    scaler = joblib.load(prepared_dir / "scaler.pkl")
    scaled = scaler.transform(df[["open","high","low","close","volume"]].values)

    assert scaled.shape == (200, 5)
