import json
from pathlib import Path

def test_model_config_valid(tmp_path):
    # Simule ton vrai model_config.json
    cfg = {
        "lookback": 30,
        "n_features": 5
    }

    path = tmp_path / "model_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f)

    loaded = json.load(open(path))

    # Vérifie la présence des clés
    assert "lookback" in loaded, "Missing `lookback` in model_config.json"
    assert "n_features" in loaded, "Missing `n_features` in model_config.json"

    # Vérifie les types
    assert isinstance(loaded["lookback"], int), "lookback must be an int"
    assert isinstance(loaded["n_features"], int), "n_features must be an int"

    # Vérifie les valeurs minimales
    assert loaded["lookback"] > 0, "lookback must be > 0"
    assert loaded["n_features"] == 5, "n_features must always be 5"
