from functools import reduce
from typing import Collection

import numpy as np
from term import Term
from utils import Point, has_duplicates

from polynomial import Polynomial


def divided_differences(x_coords: list[float], y_coords: list[float]) -> np.ndarray:
    """
    function to calculate the divided
    differences table
    """

    size = len(y_coords)
    coeffs = np.zeros([size, size])
    coeffs[:, 0] = y_coords

    for j in range(1, size):
        for i in range(size - j):
            coeffs[i][j] = (coeffs[i + 1][j - 1] - coeffs[i][j - 1]) / (x_coords[i + j] - x_coords[i])

    return coeffs


def newton_interpolation(x: list[float], y: list[float], value: float) -> float:
    if has_duplicates(x):
        raise ValueError("Interpolation requires distinct x coordinates.")

    if len(x) != len(y):
        raise ValueError("X and Y must have the same length.")

    size = len(y)
    coeffs = divided_differences(x, y)[0, :]
    p = coeffs[size - 1]

    for k in range(1, size):
        p = coeffs[size - k - 1] + (value - x[size - k - 1]) * p
    return p


def interpolate_newton(points: list[Point]) -> float:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    degree = len(points) - 1

    parameters = []

    def get_products(idx: int) -> list[list[tuple]]:
        """
        Returns:
            list[list[tuple]]: (x - x0) ... (x - xn) Produkte
        """
        return [] + [[(idx, range(idx)[i]) for i in range(x)] for x in range(idx + 1)]

    def calculate_product(indices: list[list[tuple]]) -> float:
        prod = 1
        for tpl in indices:
            a, b = tpl
            prod *= a - b
        return prod

    for n in range(degree + 1):
        products = get_products(n)
        new_parameter = points[n].y

        for i in range(n + 1):
            p = products[i]
            if i < len(parameters):
                a = parameters[i]
                new_parameter -= a * calculate_product(p)
            else:
                new_parameter /= calculate_product(p)

        parameters.append(new_parameter)

    poly = Polynomial()
    for i in range(degree + 1):
        prod = Polynomial([Term(parameters[i], 0)])
        for p in range(i):
            prod *= Term(1, 1) - Term(points[p].x, 0)

        poly += prod

    return poly


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    plt.style.use("bmh")
    x, y = [-5, -1, 0, 2], [-2, 6, 1, 3]
    x_new = np.arange(-5, 2.1, 0.1)
    y_new = newton_interpolation(x, y, x_new)

    plt.plot(x_new, y_new)
    plt.scatter(x, y)
    plt.tight_layout()
    plt.show()
