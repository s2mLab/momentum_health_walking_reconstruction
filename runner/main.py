import momentum_health_walking_reconstruction as mh
import numpy as np


def main():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = mh.adder(a, b)
    print(
        f"The result of adding {a} and {b} is {result}.\n"
        f'This is computed using momentum_health_walking_reconstruction.adder from version "{mh.__version__}"'
    )


if __name__ == "__main__":
    main()
