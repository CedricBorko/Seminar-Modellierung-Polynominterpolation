import matplotlib.pyplot as plt
import numpy as np

from aitken_neville import interpolate_aitken_neville
from lagrange import interpolate_lagrange
from utils import Point


def main() -> None:

    # lagrange_polynomial = interpolate_lagrange([Point(-1.00,0.038),Point(-0.80,0.058),Point(-0.60,0.100),Point(-0.4,0.200),Point(-0.20,0.500),Point(0.00,1.000),Point(0.20,0.500),Point(0.40,0.200),Point(0.60,0.100),Point(0.80,0.0580),Point(1.00,0.038)])
    lagrange_polynomial = interpolate_lagrange([Point(0, 1), Point(1, 3), Point(2, 6)])
    lagrange_polynomial2 = interpolate_lagrange([Point(1, 1), Point(4, 2), Point(9, 3)])
    aitken_neville_polynomial = interpolate_aitken_neville(
        [
            Point(0, 1),
            Point(1, 3),
            Point(2, 9),
        ],
        0.5,
    )
    VALUE_AT = 0.5
    # print(f"{lagrange_polynomial.value_at(VALUE_AT):.3f}")
    print(f"{aitken_neville_polynomial:.3f}")

    LEFT, RIGHT, STEP = -5, 5, 0.01

    x = np.arange(LEFT, RIGHT, STEP)
    plt.plot(x, [lagrange_polynomial.value_at(v) for v in x], label=f"{lagrange_polynomial.__repr__()}", c="#008f9b")
    plt.plot(x, [lagrange_polynomial2.value_at(v) for v in x], label=f"{lagrange_polynomial2.__repr__()}", c="#646363")
    plt.legend(fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
