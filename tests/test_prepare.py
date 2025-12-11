import pandas as pd
import numpy as np
import sys
from pathlib import Path
from src.prepare import main as prepare_main

def test_prepare_output_shapes(tmp_path, monkeypatch):

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

    # simulate CLI input: sys.argv
    monkeypatch.setattr(sys, "argv", [
        "prepare.py",
        str(raw_path),
        str(prepared_dir)
    ])

    # Call prepare.py normally
    prepare_main()

    # Check expected outputs
    assert (prepared_dir / "train").exists(), "Train dataset missing"
    assert (prepared_dir / "test").exists(), "Test dataset missing"
    assert (prepared_dir / "scaler.pkl").exists(), "Scaler missing"

