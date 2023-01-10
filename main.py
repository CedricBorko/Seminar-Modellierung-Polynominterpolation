import matplotlib.pyplot as plt
import numpy as np
from aitken_neville import interpolate_aitken_neville
from lagrange import interpolate_lagrange
from newton import interpolate_newton
from scipy import interpolate
from utils import Point

from polynomial import Polynomial


def f(x):
    x_points = [0, 1, 2, 3, 4, 5]
    y_points = [1.0, 0.5, 0.2, 0.1, 0.058823529411764705, 0.038461538461538464]

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)


def plot_graph(polynomial: Polynomial, left: float = -5, right: float = 5, step: float = 0.01, label: str = "", color: str = "#008f9b") -> None:

    x = np.arange(left, right, step)
    plt.plot(x, [polynomial.value_at(v) for v in x], label=label, c=color, linewidth=2.5)


def main() -> None:

    # # TAN(X)
    # npoly = interpolate_newton([Point(-1.5, -14.101420), Point(-0.75, -0.931596), Point(0, 0), Point(0.75, 0.931596), Point(1.5, 14.101420)])
    # lpoly = interpolate_lagrange([Point(-1.5, -14.101420), Point(-0.75, -0.931596), Point(0, 0), Point(0.75, 0.931596), Point(1.5, 14.101420)])

    fy = lambda x: 1 / (1 + x**2)

    langrange_polynomial = interpolate_lagrange(
        [Point(-5, fy(-5)), Point(-3, fy(-3)), Point(-1, fy(-1)), Point(2, fy(2)), Point(3, fy(3)), Point(5, fy(5))]
    )

    newton_polynomial = interpolate_newton(
        [Point(-5, fy(-5)), Point(-3, fy(-3)), Point(-1, fy(-1)), Point(2, fy(2)), Point(3, fy(3)), Point(5, fy(5))]
    )

    plot_graph(langrange_polynomial, color="#a51332", label="Lagrange")
    plot_graph(newton_polynomial, color="#008f9b", label="Newton")
    x = np.arange(-5, 5, 0.01)
    y = [fy(xn) for xn in x]
    # ay = [
    #     interpolate_aitken_neville([Point(-5, fy(-5)), Point(-3, fy(-3)), Point(-1, fy(-1)), Point(2, fy(2)), Point(3, fy(3)), Point(5, fy(5))], ax)
    #     for ax in x
    # ]
    # plt.plot(x, ay, label="Aitken-Neville", c="#00a331", linewidth=2.5)

    plt.plot(x, y, label="1 / (1 + x^2)", c="#646363", linewidth=2.5)

    plt.gca().set_ylim(-0.5, 2)
    plt.gca().set_xlim(-5, 5)
    plt.gca().grid()

    x_ = [-5, -3, -1, 2, 3, 5]
    y_ = [fy(a) for a in x_]
    plt.scatter(x_, y_, s=100)

    plt.legend(fancybox=True, shadow=True)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main()
