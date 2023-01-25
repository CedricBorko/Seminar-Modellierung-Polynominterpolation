import functools
import math
import operator
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt


def divided_differences_newton(
    x_coords: list[float],
    y_coords: list[float],
    previous_coefficients: np.ndarray = None
) -> np.ndarray:
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
    new_coefficients[:size - size_diff, :size - size_diff] = previous_coefficients
    new_coefficients[:, 0] = y_coords

    for j in range(1, size):
        for i in range(size - j):
            if new_coefficients[i][j] != 0: continue
            new_coefficients[i][j] = (new_coefficients[i + 1][j - 1] - new_coefficients[i][j - 1]) / (
                x_coords[i + j] - x_coords[i])

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
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (
                    x_coordinates[i + j] - x_coordinates[i])
        return table

    def __getitem__(self, indices: tuple) -> float:
        i, j = indices
        return self._divided_differences_table[i][j]

    def _shape(self) -> int:
        return len(self.x_coordinates)

    def add_point(self, xv: float, yv: float) -> None:
        self.x_coordinates.append(xv)
        self.y_coordinates.append(yv)
        self._recalculate_divided_differences()

    def _recalculate_divided_differences(self) -> None:
        new_coefficients = np.zeros((self._shape(), self._shape()))
        new_coefficients[:self._shape() - 1, :self._shape() - 1] = self._divided_differences_table
        new_coefficients[self._shape() - 1:, 0] = self.y_coordinates[-1]

        for j in range(1, self._shape()):
            for i in range(self._shape() - j):
                if new_coefficients[i][j] != 0: continue
                new_coefficients[i][j] = (new_coefficients[i + 1][j - 1] - new_coefficients[i][j - 1]) / (
                    self.x_coordinates[i + j] - self.x_coordinates[i])
        self._divided_differences_table = new_coefficients

    @property
    def divided_differences_table(self):
        return self._divided_differences_table

    def coefficients(self) -> tuple[float]:
        return tuple(self._divided_differences_table[0, :])

    def __call__(self, value: float) -> float:
        return sum(
            (self.coefficients()[i] *
             functools.reduce(operator.mul, [(value - self.x_coordinates[j]) for j in range(i)], 1))
            for i in range(self._shape())
        )

    def add_points(self, x_coordinates: list[float], y_coordinates: list[float]) -> None:
        for x_, y_ in zip(x_coordinates, y_coordinates):
            self.add_point(x_, y_)

    def plot(
        self,
        step: float = 0.01,
        label: str = "Newton Interpolation",
        func: Callable[[float], float] = None,
        xmin: float = None,
        xmax: float = None
    ) -> None:
        plt.style.use("fivethirtyeight")
        if all((xmin, xmax)):
            plt.gca().set_xlim(xmin, xmax)
            xr = np.arange(xmin, xmax + step, step)
        else:
            xr = np.arange(-1 * abs(max(self.x_coordinates)), abs(max(self.x_coordinates)) + step, step)

        plt.plot(xr, [self(xv) for xv in xr], label=f"{label} {self._shape()} Punkte", lw=6)
        if func:
            plt.plot(xr, [func(xv) for xv in xr], label=f"{func.__name__}", lw=3)
        plt.gca().set_ylim(-2, 2)

        plt.legend()


if __name__ == '__main__':
    # x = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    # y = [math.cos(xv) for xv in x]
    # n = NewtonPolynomial(x, y)
    # n.add_points([math.pi / 8, math.pi / 12], [math.cos(math.pi / 8), math.cos(math.pi / 12)])
    # n.plot(label="cos", func=math.cos, xmin=-math.pi, xmax=math.pi)
    np.set_printoptions(suppress=True)

    x = [0, 1, 2]
    y = [0, 1, 4]
    n2 = NewtonPolynomial(x, y)
    # n2.add_points([math.pi / 8, math.pi / 12], [math.sin(math.pi / 8), math.sin(math.pi / 12)])
    print(n2.divided_differences_table)
    n2.add_point(3, 6)
    print(n2.divided_differences_table)
    n2.add_point(4, -5)
    print(n2.divided_differences_table)
