import operator
import random
from enum import Enum
from functools import reduce

import numpy as np

from polynomial import Polynomial


def has_duplicates(x_coordinates: list[float]) -> bool:
    return len(list(x_coordinates)) != len(set(x_coordinates))


def is_valid_input(x_coords: list[float], y_coords: list[float]) -> bool:
    return not has_duplicates(x_coords) and len(x_coords) == len(y_coords)


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


def interpolate_newton(x_coords: list[float], y_coords: list[float], value: float = None) -> float:
    if not is_valid_input(x_coords, y_coords):
        raise ValueError("Invalid Input. X coordinates must be distinct and x, y must have the same length.")
    # f0 = P(x0) = a0
    # f1 = P(x1) = a0 + a1(x1 - x0)
    # P(x) = a0 + a1(x - x0) + a2(x - x0)(x - x1) + ... + an(x - x0) ... (x - xn)

    size = len(x_coords)
    coeffs = divided_differences(x_coords, y_coords)[0, :]
    polynomial = coeffs[0]

    for n in range(1, size):
        product = reduce(operator.mul, (Polynomial({1: 1, 0: -x_coords[k]}) for k in range(n)))
        polynomial = product * coeffs[n] + polynomial
    return polynomial(value) if value is not None else polynomial


def interpolate_aitken_neville(x_coords: list[float], y_coords: list[float], value: float) -> float:
    if not is_valid_input(x_coords, y_coords):
        raise ValueError("Invalid Input. X coordinates must be distinct and x, y must have the same length.")

    if value in x_coords:
        return y_coords[x_coords.index(value)]

    size = len(x_coords)
    polynomials = [y_coords[i] for i in range(size)]

    # P[i, j](x) = ((x - xi) * P[i + 1, j](x) - (x - xj) * P[i, j - 1](x)) / (xj - xi)

    # E.g. x_coords = [1, 2, 3]; y_coords = [1, 4, 9]; x = 5
    # P[n, n] = y_coords[n]
    # Solution is P[0, n - 1]

    # P[0, 1] = ((x - x0) * P[1, 1] - (x - x1) * P[0, 0]) / (x1 - x0)
    # P[0, 1] = ((5 - 1) * P[1, 1] - (5 - 2) * P[0, 0]) / (2 - 1)
    # P[0, 1] = ((4) * y_coords[1] - (3) * y_coords[0]) / 1
    # P[0, 1] = (4 * 4 - 3 * 1)
    # P[0, 1] = 16 - 3 = 13

    # P[1, 2] = ((x - x1) * P[2, 2] - (x - x2) * P[1, 1]) / (x2 - x1)
    # P[1, 2] = ((5 - 2) * P[2, 2] - (5 - 3) * P[1, 1]) / (3 - 2)
    # P[1, 2] = ((3) * y_coords[2] - (2) * y_coords[1]) / 1
    # P[1, 2] = (3 * 9 - 2 * 4)
    # P[1, 2] = 27 - 8 = 19

    # P[0, 2] = ((x - x0) * P[1, 2] - (x - x2) * P[0, 1]) / (x2 - x0)
    # P[0, 2] = ((5 - 1) * P[1, 2] - (5 - 3) * P[0, 1]) / (3 - 1)
    # P[0, 2] = ((4) * 19 - (2) * 13) / 2
    # P[0, 2] = (76 - 26) / 2
    # == P[0, 2] = 25 ==

    for j in range(1, size):
        for i in range(size - j):
            numerator = (value - x_coords[i + j]) * polynomials[i] + (x_coords[i] - value) * polynomials[i + 1]
            denominator = x_coords[i] - x_coords[i + j]
            polynomials[i] = numerator / denominator

    return polynomials[0]


def interpolate_lagrange(x_coords: list[float], y_coords: list[float], value: float = None) -> Polynomial:
    if has_duplicates(x_coords):
        raise ValueError("Interpolation requires distinct x coordinates.")

    if len(x_coords) != len(y_coords):
        raise ValueError("X and Y must have the same length.")

    size = len(x_coords)

    lagrange_polynomials = [1] * size
    for i in range(size):
        for j in range(size):
            if j == i:
                continue

            numerator = Polynomial({1: 1}) - x_coords[j]
            denominator = x_coords[i] - x_coords[j]
            lagrange_polynomials[i] = numerator / denominator * lagrange_polynomials[i]

    sum_poly = Polynomial()
    for i, poly in enumerate(lagrange_polynomials):
        sum_poly += poly * y_coords[i]
    return sum_poly(value) if value is not None else sum_poly


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    L = interpolate_lagrange([1, 2, 3, 4], [0, -42, -80, -108])
    L2 = interpolate_lagrange([1, 2, -1], [4, 9, 6])
    L3 = interpolate_lagrange([-1, 0, 1], [1, 3, -7])

    def getY(poly: Polynomial, __x: np.arange) -> list:
        return [poly(_x) for _x in __x]

    plt.style.use("bmh")
    x = np.arange(-10, 10, 0.01)
    plt.plot(x, getY(L, x), label=f"L(x)  = {L.__repr__()}")
    plt.plot(x, getY(L2, x), label=f"L2(x) = {L2.__repr__()}")
    plt.plot(x, getY(L3, x), label=f"L3(x) = {L3.__repr__()}")

    plt.legend()
    plt.tight_layout()
    plt.show()
