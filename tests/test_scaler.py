import numpy as np
from sklearn.preprocessing import MinMaxScaler

def test_scaler_inverse_correct():
    data = np.array([
        [10,20,30,40,50],
        [20,30,40,50,60],
    ])

    scaler = MinMaxScaler()
    scaler.fit(data)

    X = scaler.transform(data)
    X_inv = scaler.inverse_transform(X)

    assert np.allclose(data, X_inv), "Inverse transform does not match original"
