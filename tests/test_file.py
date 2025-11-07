import numpy as np
import mhomentum_health_walking_reconstruction as mh


def test_version():
    assert mh.__version__ == "0.1.0"


def test_adder():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = mh.adder(a, b)
    assert np.all(result == np.array([5, 7, 9]))
