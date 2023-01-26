from matplotlib import patheffects

from interpolation import timing
from polynomial import Polynomial, SUB

import math
from matplotlib import pyplot as plt
from typing import Callable, Iterable
import numpy as np


def generate_x_coordinates_for_interpolation(amount: int = 5) -> list[int]:
    return [x - amount // 2 for x in range(amount)]


def generate_y_coordinates(
    x_coordinates: Iterable,
    func: Callable[[int | float], float],
    fill_value: float = 0
) -> list[float]:
    y_coordinates = []
    for x in x_coordinates:
        try:
            y_coordinates.append(func(x))
        except ValueError:
            y_coordinates.append(fill_value)
    return y_coordinates


def interpolation_lagrange(x_coordinates: list[float], y_coordinates: list[float]) -> Polynomial:
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("X and Y must have the same length.")

    size = len(x_coordinates)

    lagrange_polynomials = [Polynomial.one()] * size
    for i in range(size):
        for j in range(size):
            if j == i:
                continue
            numerator = Polynomial.x() - x_coordinates[j]
            denominator = x_coordinates[i] - x_coordinates[j]
            lagrange_polynomials[i] = numerator / denominator * lagrange_polynomials[i]

    return sum(polynomial * y_value for polynomial, y_value in zip(lagrange_polynomials, y_coordinates))


def get_lagrange_polynomials(x_coordinates: list[float], y_coordinates: list[float]) -> list[
    Polynomial]:
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("X and Y must have the same length.")

    size = len(x_coordinates)

    lagrange_polynomials = [Polynomial.one()] * size
    for i in range(size):
        for j in range(size):
            if j == i:
                continue

            numerator = Polynomial.x() - x_coordinates[j]
            denominator = x_coordinates[i] - x_coordinates[j]
            lagrange_polynomials[i] = numerator / denominator * lagrange_polynomials[i]

    return lagrange_polynomials


def get_lagrange_polynomials_str(x_coordinates: list[float], y_coordinates: list[float]) -> list[str]:
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("X and Y must have the same length.")

    size = len(x_coordinates)

    lagrange_polynomials = [""] * size
    for i in range(size):
        n = []
        d = []
        for j in range(size):
            if j == i: continue

            n.append(f"(x {'-' if x_coordinates[j] > 0 else '+'} {abs(x_coordinates[j])})")
            d.append(f"({x_coordinates[i]} {'-' if x_coordinates[j] > 0 else '+'} {abs(x_coordinates[j])})")

        lagrange_polynomials[i] = ''.join(n) + "\n" + ("-" * sum(len(d_) for d_ in d)) + "\n" + ''.join(d)
    return lagrange_polynomials


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def main():
    plt.style.use("bmh")

    plt.rcParams["font.size"] = 14
    func = math.sin
    N_POINTS: int = 3
    MAX_POINTS: int = 50
    STEP: float = 0.01
    Y_LIMIT = (-2, 5)

    x_coords_interpolation = [1, 2, 3]
    y_coords_interpolation = [3, 1, 4]

    for p in get_lagrange_polynomials_str(x_coords_interpolation, y_coords_interpolation):
        print(p)
        print()

    x_smooth = np.arange(-MAX_POINTS // 2, MAX_POINTS // 2 + STEP, STEP)
    y_smooth = generate_y_coordinates(x_smooth, func, fill_value=np.nan)
    polynomial = interpolation_lagrange(x_coords_interpolation, y_coords_interpolation)

    p = get_lagrange_polynomials(x_coords_interpolation, y_coords_interpolation)
    figure, axes = plt.subplots()
    axes.set_facecolor("#fff")
    axes.set_ylim(Y_LIMIT)
    axes.set_xlim(0, 5)
    axes.set_title(f"Interpolation mit {len(x_coords_interpolation)} Punkten.")
    mul_y = False

    for idx in range(3):
        y_interpolated = generate_y_coordinates(
            x_smooth,
            p[idx] * (y_coords_interpolation[idx] if mul_y else 1),
            fill_value=np.nan
        )

        axes.plot(
            x_smooth,
            y_interpolated,
            label=f"L{idx}(x)".translate(SUB) + " = " + (
                p[idx] * (y_coords_interpolation[idx] if mul_y else 1)).__repr__(),
            lw=3
        )

        # scatter = plt.scatter(
        #     [xv for xv in x_coords_interpolation[:idx + 1]] * (2 if idx == 2 else 1),
        #     [1 for _ in x_coords_interpolation[:idx + 1]] + (
        #         [0 for _ in x_coords_interpolation[:idx + 1]] if idx == 2 else []),
        #     zorder=4,
        #     s=160,
        #     facecolor="none",
        #     edgecolor="k",
        #     linewidth=2
        # )

    # axes.plot(
    #     x_smooth,
    #     generate_y_coordinates(x_smooth, polynomial, fill_value=np.nan),
    #     c="r",
    #     label="P(x)" + " = " + polynomial.__repr__(),
    #     lw=2.5
    # )

    # scatter = plt.scatter(
    #     x_coords_interpolation,
    #     y_coords_interpolation,
    #     zorder=4,
    #     s=120,
    #     facecolor="r",
    # )
    axes.legend(fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()
    #
    # figure, axes = plt.subplots()
    #
    # y_interpolated = generate_y_coordinates(x_smooth, polynomial, fill_value=np.nan)
    #
    # function_line, = axes.plot(x_smooth, y_smooth, c="k", label=func.__name__, lw=2.5)
    # interpolated_line, = axes.plot(x_smooth, y_interpolated, c="r", label="Interpolation", lw=2.5)
    #
    # lagrange_lines = []
    # for idx, lagrange_polynomial in enumerate(
    #     get_lagrange_polynomials(x_coords_interpolation, y_coords_interpolation)
    # ):
    #     lagrange_line, = axes.plot(
    #         x_smooth,
    #         generate_y_coordinates(x_smooth, lagrange_polynomial, fill_value=np.nan),
    #         alpha=0.75,
    #         label=f"L[{idx}]"
    #     )
    #     lagrange_lines.append(lagrange_line)
    #
    # scatter = plt.scatter(
    #     [xv for xv in x_coords_interpolation] * 2,
    #     [0 for _ in x_coords_interpolation] + [1 for _ in x_coords_interpolation],
    #     zorder=4,
    #     s=160,
    #     facecolor="none",
    #     edgecolor="k",
    #     linewidth=2
    # )
    #
    # axes.set_title(f"Interpolation von y = {func.__name__}(x) mit {N_POINTS} Punkten.")
    #
    # axes.set_ylim(Y_LIMIT)
    # axes.set_xlim(-N_POINTS // 2 - 1, abs(-N_POINTS // 2 - 1))
    #
    # axes.legend()
    # figure.tight_layout()
    #
    # def on_mouse_wheel(event: MouseEvent) -> None:
    #     delta = 1 if event.button == "up" else -1
    #     nonlocal N_POINTS, function_line, scatter, interpolated_line, x_coords_interpolation, y_coords_interpolation, x_smooth, y_smooth, polynomial, y_interpolated
    #
    #     N_POINTS += delta
    #     N_POINTS = clamp(N_POINTS, 2, MAX_POINTS)
    #
    #     x_coords_interpolation = generate_x_coordinates_for_interpolation(N_POINTS)
    #     y_coords_interpolation = generate_y_coordinates(x_coords_interpolation, func)
    #
    #     x_smooth = np.arange(-MAX_POINTS // 2, MAX_POINTS // 2 + STEP, STEP)
    #     y_smooth = generate_y_coordinates(x_smooth, func, fill_value=np.nan)
    #     polynomial = interpolation_lagrange(x_coords_interpolation, y_coords_interpolation)
    #     y_interpolated = generate_y_coordinates(x_smooth, polynomial, fill_value=np.nan)
    #
    #     if function_line:
    #         function_line.remove()
    #         interpolated_line.remove()
    #
    #     for line in lagrange_lines:
    #         line.remove()
    #
    #     lagrange_lines.clear()
    #     for idx, lagrange_polynomial in enumerate(
    #         get_lagrange_polynomials(x_coords_interpolation, y_coords_interpolation)
    #     ):
    #         lagrange_line, = axes.plot(
    #             x_smooth,
    #             generate_y_coordinates(x_smooth, lagrange_polynomial, fill_value=np.nan),
    #             alpha=0.75,
    #             label=f"L[{idx}]"
    #         )
    #         lagrange_lines.append(lagrange_line)
    #
    #     if scatter:
    #         scatter.remove()
    #
    #     scatter = plt.scatter(
    #         [xv for xv in x_coords_interpolation] * 2,
    #         [0 for _ in x_coords_interpolation] + [1 for _ in x_coords_interpolation],
    #         zorder=4,
    #         s=160,
    #         facecolor="none",
    #         edgecolor="k",
    #         linewidth=2
    #     )
    #
    #     function_line, = axes.plot(x_smooth, y_smooth, c="k", label=func.__name__, lw=2.5)
    #     interpolated_line, = axes.plot(x_smooth, y_interpolated, c="r", label="Interpolation", lw=2.5)
    #     axes.set_title(f"Interpolation von y = {func.__name__}(x) mit {N_POINTS} Punkten.")
    #
    #     axes.set_ylim(Y_LIMIT)
    #     axes.set_xlim(-N_POINTS // 2 - 1, abs(-N_POINTS // 2 - 1))
    #
    #     axes.legend()
    #     figure.tight_layout()
    #     figure.canvas.draw()
    #
    # figure.canvas.mpl_connect('scroll_event', on_mouse_wheel)
    # plt.show()


@timing
def test_lagrange() -> None:
    data = [(-1.0, 0.038461538461538464), (-0.9333333333333333, 0.04390243902439024),
            (-0.8666666666666667, 0.05056179775280899), (-0.8, 0.05882352941176469),
            (-0.7333333333333334, 0.0692307692307692), (-0.6666666666666667, 0.08256880733944953),
            (-0.6000000000000001, 0.09999999999999998), (-0.5333333333333334, 0.12328767123287666),
            (-0.4666666666666668, 0.15517241379310337), (-0.40000000000000013, 0.1999999999999999),
            (-0.3333333333333335, 0.264705882352941), (-0.26666666666666683, 0.3599999999999997),
            (-0.20000000000000018, 0.49999999999999956), (-0.13333333333333353, 0.6923076923076917),
            (-0.06666666666666687, 0.8999999999999995), (-2.220446049250313e-16, 1.0),
            (0.06666666666666643, 0.9000000000000007), (0.13333333333333308, 0.6923076923076931),
            (0.19999999999999973, 0.5000000000000007), (0.2666666666666664, 0.3600000000000005),
            (0.33333333333333304, 0.2647058823529415), (0.3999999999999997, 0.20000000000000026),
            (0.46666666666666634, 0.15517241379310362), (0.533333333333333, 0.12328767123287686),
            (0.5999999999999996, 0.1000000000000001), (0.6666666666666663, 0.08256880733944962),
            (0.733333333333333, 0.0692307692307693), (0.7999999999999996, 0.058823529411764754),
            (0.8666666666666663, 0.05056179775280903), (0.9333333333333329, 0.04390243902439028)]

    plt.style.use("bmh")
    x, y = zip(*(data_point for data_point in data))
    p = interpolation_lagrange(x, y)
    STEP = 0.01
    x_smooth = np.arange(-math.floor(abs(max(x))), math.ceil(abs(max(x))), STEP)
    y_interpolated = generate_y_coordinates(
        x_smooth,
        p,
        fill_value=np.nan
    )

    plt.plot(
        x_smooth,
        y_interpolated,
        label=f"P(x) = " + p.__repr__(),
        lw=3
    )

    plt.tight_layout()


if __name__ == '__main__':
    main()
