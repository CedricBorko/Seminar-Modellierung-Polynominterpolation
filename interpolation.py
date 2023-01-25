import math
import operator
from functools import reduce
from typing import Callable

import math
import sympy as sp
import numpy as np

from polynomial import Polynomial
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return wrap


@timing
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


def matrix_to_polynom(matrix):
    x = sp.Symbol('x')
    f = sp.Function('z')(x)
    f += matrix[0][1]
    size = len(matrix)
    for i in range(2, size + 1):
        zeroing = 1
        for a in range(i - 1):
            zeroing = zeroing * (x - matrix[a][0])
        poly = matrix[0][i] * zeroing
        f += poly
    f -= sp.Function('z')(x)
    return sp.simplify(f)


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


def interpolate_newton(
    x_coords: list[float],
    y_coords: list[float],
    value: float = None
) -> float | Polynomial:
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
            numerator = (value - x_coords[i + j]) * polynomials[i] + (x_coords[i] - value) * polynomials[
                i + 1]
            denominator = x_coords[i] - x_coords[i + j]
            polynomials[i] = numerator / denominator

    return polynomials[0]


def interpolate_lagrange(
    x_coords: list[float],
    y_coords: list[float],
    value: float = None,
    get_polynomials: bool = False
) -> \
    list[float] | float | Polynomial:
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

    if get_polynomials: return [p * y for y, p in zip(y_coords, lagrange_polynomials)]
    sum_poly = sum(polynomial * y_value for polynomial, y_value in zip(lagrange_polynomials, y_coords))
    return sum_poly(value) if value is not None else sum_poly


def interpolate_nevillePoint(p, xlist, ylist, point):
    x, y = xlist, ylist
    size = len(x)
    diff = size - len(p)
    if diff == 0:
        for k in range(1, size):
            for i in range(size):
                if i + k <= size - 1:  # 0 <= i <= i + k <= n)
                    p[i][k] = ((point - x[i]) * p[i + 1][k - 1]
                               - (point - x[i + k]) * p[i][k - 1]) / (x[i + k] - x[i])  # (3.20)
    else:
        for b in range(len(p)):
            p[b] = p[b] + [0] * diff
        for a in range(diff):
            p.append([0] * size)
        for c in range((size - diff), size):
            p[c][0] = y[c]
        for k in range(1, size):
            for i in range(size):
                if i + k <= size - 1:  # 0 <= i <= i + k <= n)
                    if p[i][k] == 0:
                        p[i][k] = ((point - x[i]) * p[i + 1][k - 1]
                                   - (point - x[i + k]) * p[i][k - 1]) / (x[i + k] - x[i])  # (3.20)
    return p


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # x_, y_ = [1, 2, 4], [4, 5, 1]
    # polys = interpolate_lagrange(x_, y_, get_polynomials=True)
    #
    # x = np.arange(-MAX_POINTS // 2, MAX_POINTS // 2, 0.01)
    # plt.style.use("bmh")
    # plt.rcParams["font.size"] = "20"
    #
    #
    # def getY(poly: Callable, __x: np.arange) -> list:
    #     y__ = []
    #     for i in __x:
    #         try:
    #             y__.append(poly(i))
    #         except ValueError:
    #             y__.append(0)
    #     return y__
    #
    # for idx, poly in enumerate(polys):
    #     plt.plot(x, getY(poly, x), label=f"{x_[idx], y_[idx]}: {poly.__repr__()}", zorder=2)
    #     plt.annotate(f"{x_[idx], y_[idx]}", (x_[idx], y_[idx] + 1))
    # plt.scatter(x_, y_, s=50, zorder=3, color="k")
    # plt.scatter(x_, [0] * len(x_), s=100, zorder=3, color="r")
    #
    # # for xv in x_:
    # #     plt.gca().axvline(xv, c="k", zorder=1, alpha=0.5)
    #
    # s_poly = sum(polys)
    # # plt.plot(x, getY(s_poly, x), label="L(x) = " + s_poly.__repr__())
    # plt.legend()
    # plt.gca().set_xticks(range(-4, 9))
    # plt.gca().set_yticks(range(-10, 11))
    # plt.gca().set_xlim(-4, 8)
    # plt.gca().set_ylim(-10, 10)
    #
    # plt.tight_layout()
    # plt.show()

    MAX_POINTS = 30
    points = 5

    F_1 = math.exp
    F_2 = math.sin


    def calculate_interpolation(n_points: int) -> tuple[Polynomial, Polynomial]:
        x__ = [n for n in range(-n_points // 2 + 1, n_points // 2 + 1)]
        y__ = []
        for i in x__:
            try:
                y__.append(F_1(i))
            except ValueError:
                y__.append(0)
        F1_ = interpolate_lagrange(x__, y__)
        y__ = []
        for i in x__:
            try:
                y__.append(F_2(i))
            except ValueError:
                y__.append(0)
        F2_ = interpolate_newton(x__, y__)
        return F1_, F2_


    def getY(poly: Callable, __x: np.arange) -> list:
        y__ = []
        for i in __x:
            try:
                y__.append(poly(i))
            except ValueError:
                y__.append(0)
        return y__


    plt.style.use("bmh")
    figure, axes = plt.subplots(figsize=(16, 9))
    F1, F2 = calculate_interpolation(points)
    x = np.arange(-MAX_POINTS // 2 - 1, abs(-MAX_POINTS // 2 - 1), 0.01)
    axes.set_xlim(-MAX_POINTS // 2 - 1, abs(-MAX_POINTS // 2 - 1))
    axes.set_ylim(-4, 4)
    axes.set_title(f"Interpolation mit {points} Punkten.")

    line1, = axes.plot(x, getY(F_1, x), label=F_1.__name__, lw=4, alpha=0.25, c="g")
    # line2, = axes.plot(x, getY(F_2, x), label=F_2.__name__, lw=4, alpha=0.25, c="g")

    x__ = [n for n in range(-points // 2 + 1, points // 2 + 1)]
    y__ = getY(F_1, x__)
    scatter = plt.scatter(x__, y__, s=40, c="k", zorder=5)

    line3, = axes.plot(x, getY(F1, x), label=f"{F1.__repr__()}", c="g")
    # line4, = axes.plot(x, getY(F2, x), label=f"{F2.__repr__()}", c="g")
    legend = axes.legend(shadow=True, fancybox=True, loc=0)
    figure.tight_layout()

    lines = [line1, line3]
    lined = {}  # Will map legend lines to original lines.
    for legline, origline in zip(legend.get_lines(), lines):
        legline.set_picker(True)  # Enable picking on the legend line.
        lined[legline] = origline


    def on_pick(event) -> None:
        legend_line = event.artist
        original_line = lined[legend_line]
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        figure.canvas.draw()


    def on_press(event) -> None:
        global points, line3, F1, F2, figure, legend, lined, x, line1, lines, scatter

        increment = 1 if event.button == 'up' else -1
        points += increment
        if points == 1:
            points = 2

        if points > MAX_POINTS:
            points = MAX_POINTS

        try:
            line1.remove()
            line3.remove()
            F1, F2 = calculate_interpolation(points)
            x = np.arange(-MAX_POINTS // 2 - 1, abs(-MAX_POINTS // 2 - 1), 0.01)
            axes.set_xlim(-MAX_POINTS // 2 - 1, abs(-MAX_POINTS // 2 - 1))
            axes.set_ylim(-4, 4)

            visible = []
            for legline, origline in zip(legend.get_lines(), lines):
                legline.set_picker(True)  # Enable picking on the legend line.
                lined[legline] = origline
                visible.append(lined[legline].get_visible())

            line1, = axes.plot(x, getY(F_1, x), label=F_1.__name__, lw=4, alpha=0.25, c="g")
            # line2, = axes.plot(x, getY(F_2, x), label=F_2.__name__, lw=4, alpha=0.25, c="g")

            line3, = axes.plot(x, getY(F1, x), label=f"{F1.__repr__()}", c="g")
            # line4, = axes.plot(x, getY(F2, x), label=f"{F2.__repr__()}", c="g")

            if scatter:
                scatter.remove()
            x_ = [n for n in range(-points // 2 + 1, points // 2 + 1)]
            y_ = getY(F_1, x_)
            scatter = plt.scatter(x_, y_, s=40, c="k", zorder=5)

            line1.set_visible(visible[0])
            # line2.set_visible(visible[1])
            line3.set_visible(visible[1])
            # line4.set_visible(visible[3])

            if legend:
                legend.remove()

            legend = axes.legend(shadow=True, fancybox=True, loc=9)
            axes.set_title(f"Interpolation mit {points} Punkten.")
            lines = [line1, line3]
            lined = {}  # Will map legend lines to original lines.
            for legline, origline in zip(legend.get_lines(), lines):
                legline.set_picker(True)  # Enable picking on the legend line.
                lined[legline] = origline

            figure.tight_layout()
            figure.canvas.draw()
        except ValueError:
            pass


    figure.canvas.mpl_connect('pick_event', on_pick)
    figure.canvas.mpl_connect('scroll_event', on_press)
    plt.show()
