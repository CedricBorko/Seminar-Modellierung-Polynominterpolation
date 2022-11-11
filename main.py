import matplotlib.pyplot as plt
import numpy as np

from aitken_neville import interpolate_aitken_neville
from lagrange import interpolate_lagrange
from newton import interpolate_newton
from polynomial import Polynomial
from utils import Point


def plot_graph(polynomial: Polynomial, left: float = -10, right: float = 10, step: float = 0.01, color: str = "#008f9b") -> None:

    x = np.arange(left, right, step)
    plt.plot(x, [polynomial.value_at(v) for v in x], label=f"{polynomial.__repr__()}", c=color)


def main() -> None:

    # TAN(X)
    npoly = interpolate_newton([Point(-1.5, -14.101420), Point(-0.75, -0.931596), Point(0, 0), Point(0.75, 0.931596), Point(1.5, 14.101420)])
    lpoly = interpolate_lagrange([Point(-1.5, -14.101420), Point(-0.75, -0.931596), Point(0, 0), Point(0.75, 0.931596), Point(1.5, 14.101420)])

    plot_graph(npoly, color="#646363")
    plot_graph(lpoly, color="#a51332")

    plt.legend(fancybox=True, shadow=True)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main()
