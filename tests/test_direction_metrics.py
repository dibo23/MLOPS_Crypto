import numpy as np
from src.evaluate import compute_direction

def test_compute_direction_basic():
    arr = np.array([1,2,1,5,5])
    dir = compute_direction(arr)

    assert len(dir) == 4
    assert list(dir) == [1, -1, 1, 1]  # up, down, up, flat(up)
