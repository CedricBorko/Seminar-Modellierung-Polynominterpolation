import functools
import math
import operator
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from interpolation import interpolate_aitken_neville, interpolate_nevillePoint
from main import generate_y_coordinates
from polynomial import SUB, Polynomial


def divided_differences_newton(x_coords: list[float], y_coords: list[float], previous_coefficients: np.ndarray = None) -> np.ndarray:
    size = len(x_coords)
    if previous_coefficients is None:
        coeffs = np.zeros([size, size])
        coeffs[:, 0] = y_coords

        for j in range(1, size):
            for i in range(size - j):
                coeffs[i][j] = (coeffs[i + 1][j - 1] - coeffs[i][j - 1]) / (x_coords[i + j] - x_coords[i])

        return coeffs

    size_diff = size - previous_coefficients.shape[0]
    new_coefficients = np.zeros((size, size))
    new_coefficients[: size - size_diff, : size - size_diff] = previous_coefficients
    new_coefficients[:, 0] = y_coords

    for j in range(1, size):
        for i in range(size - j):
            if new_coefficients[i][j] != 0:
                continue
            new_coefficients[i][j] = (new_coefficients[i + 1][j - 1] - new_coefficients[i][j - 1]) / (x_coords[i + j] - x_coords[i])

    return new_coefficients


class NewtonPolynomial:
    def __init__(self, x_coordinates: list[float], y_coordinates: list[float]) -> None:
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates

        self._divided_differences_table = self.calculate_divided_differences(x_coordinates, y_coordinates)

    @staticmethod
    def calculate_divided_differences(x_coordinates: list[float], y_coordinates: list[float]) -> np.ndarray:
        size = len(x_coordinates)
        table = np.zeros(shape=(size, size))
        table[:, 0] = y_coordinates
        for j in range(1, size):
            for i in range(size - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_coordinates[i + j] - x_coordinates[i])
        return table

    def __getitem__(self, indices: tuple) -> float:
        i, j = indices
        return self._divided_differences_table[i][j]

    def shape(self) -> int:
        return len(self.x_coordinates)

    def add_point(self, xv: float, yv: float) -> None:
        if xv in self.x_coordinates:
            raise ValueError("Interpolation points must be distinct.")
        self.x_coordinates.append(xv)
        self.y_coordinates.append(yv)
        self._recalculate_divided_differences()

    def _recalculate_divided_differences(self) -> None:
        new_coefficients = np.zeros((self.shape(), self.shape()))
        new_coefficients[: self.shape() - 1, : self.shape() - 1] = self._divided_differences_table
        new_coefficients[self.shape() - 1 :, 0] = self.y_coordinates[-1]

        for j in range(1, self.shape()):
            for i in range(self.shape() - j):
                if new_coefficients[i][j] != 0:
                    continue
                new_coefficients[i][j] = (new_coefficients[i + 1][j - 1] - new_coefficients[i][j - 1]) / (
                    self.x_coordinates[i + j] - self.x_coordinates[i]
                )
        self._divided_differences_table = new_coefficients

    @property
    def divided_differences_table(self):
        return self._divided_differences_table

    def coefficients(self) -> tuple[float]:
        return tuple(self._divided_differences_table[0, :])

    def __call__(self, value: float) -> float:
        return sum(
            (self.coefficients()[i] * functools.reduce(operator.mul, [(value - self.x_coordinates[j]) for j in range(i)], 1))
            for i in range(self.shape())
        )

    def add_points(self, x_coordinates: list[float], y_coordinates: list[float]) -> None:
        for x_, y_ in zip(x_coordinates, y_coordinates):
            self.add_point(x_, y_)

    def as_polynomial(self) -> Polynomial:
        polynomial = self.coefficients()[0]

        for n in range(1, self.shape()):
            product = functools.reduce(operator.mul, ((Polynomial.x() - self.x_coordinates[k]) for k in range(n)))
            polynomial = product * self.coefficients()[n] + polynomial
        return polynomial


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    x = [1, 2, 3]
    y = [3, 1, 4]
    n2 = NewtonPolynomial(x, y)
    plt.style.use("bmh")
    plt.rcParams["font.size"] = 14

    figure, axes = plt.subplots(1, 1, figsize=(16, 9))

    axes.set_facecolor("#fff")
    axes.set_xlim(0, 5)
    axes.set_ylim(-2, 16)
    axes.set_ylabel("P(x)")
    axes.set_xlabel("x")
    axes.set_title("Newton Interpolation")

    x_smooth = np.linspace(0, 5, 50)

    func = NewtonPolynomial(x, y)
    axes.plot(x_smooth, generate_y_coordinates(x_smooth, func), label="P2(x) = ".translate(SUB) + func.as_polynomial().__repr__())

    func2 = NewtonPolynomial(x + [4], y + [6])
    axes.plot(x_smooth, generate_y_coordinates(x_smooth, func2), label="P3(x) = ".translate(SUB) + func2.as_polynomial().__repr__())

    axes.scatter(x, y, c="r", s=100, zorder=3)
    axes.legend(fancybox=True, shadow=True, loc=9)
    plt.tight_layout()
    plt.show()
