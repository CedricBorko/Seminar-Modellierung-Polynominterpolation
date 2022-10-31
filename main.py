

import matplotlib.pyplot as plt
import numpy as np

from data import Point, Polynomial, Term
from lagrange import interpolate_lagrange


def main() -> None:

    lagrange_polynomial = interpolate_lagrange([Point(-1.00,0.038),Point(-0.80,0.058),Point(-0.60,0.100),Point(-0.4,0.200),Point(-0.20,0.500),Point(0.00,1.000),Point(0.20,0.500),Point(0.40,0.200),Point(0.60,0.100),Point(0.80,0.0580),Point(1.00,0.038)])
    VALUE_AT = 0.5
    print(f"{lagrange_polynomial.value_at(VALUE_AT):.3f}")

    LEFT, RIGHT, STEP = -1, 1, 0.001


    x = np.arange(LEFT, RIGHT, STEP)
    plt.plot(x, [lagrange_polynomial.value_at(v) for v in x], label=f"{lagrange_polynomial.__repr__()}", c="#008f9b")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
