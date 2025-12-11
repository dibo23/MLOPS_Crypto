import json
from pathlib import Path

def test_model_config_structure(tmp_path):
    cfg = {
        "lookback": 30,
        "n_features": 5
    }

    path = tmp_path / "model_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f)

    loaded = json.load(open(path))

    assert "lookback" in loaded, "Missing `lookback` in model_config"
    assert "n_features" in loaded, "Missing `n_features` in model_config"
    assert isinstance(loaded["lookback"], int)
    assert loaded["n_features"] == 5
