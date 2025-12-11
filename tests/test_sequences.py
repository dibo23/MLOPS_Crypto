import numpy as np
import pandas as pd
from src.evaluate import create_sequences


def test_create_sequences_shapes():
    df = pd.DataFrame({
        "open": [1,2,3,4,5],
        "high": [1,2,3,4,5],
        "low": [1,2,3,4,5],
        "close": [1,2,3,4,5],
        "volume": [1,2,3,4,5],
    })

    X, y = create_sequences(df, lookback=2)

    assert X.shape == (3, 2, 5), "Wrong X shape"
    assert y.shape == (3,), "Wrong y shape"
    assert np.allclose(y, [3,4,5]), "Wrong target values"


def test_sequence_performance():
    import time

    df = pd.DataFrame({
        c: range(5000) for c in ["open", "high", "low", "close", "volume"]
    })

    t0 = time.time()
    X, y = create_sequences(df, lookback=30)
    dt = time.time() - t0

    # Séquence de 5000 lignes → doit être rapide
    assert dt < 0.5, f"create_sequences too slow ({dt:.3f}s)"
