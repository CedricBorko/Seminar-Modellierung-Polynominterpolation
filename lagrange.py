import numpy as np
from matplotlib import pyplot as plt

from main import generate_y_coordinates
from polynomial import SUB, SUP, Polynomial


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


def get_lagrange_polynomials(x_coordinates: list[float], y_coordinates: list[float]) -> list[Polynomial]:
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
            if j == i:
                continue

            n.append(f"(x {'-' if x_coordinates[j] > 0 else '+'} {abs(x_coordinates[j])})")
            d.append(f"({x_coordinates[i]} {'-' if x_coordinates[j] > 0 else '+'} {abs(x_coordinates[j])})")

        lagrange_polynomials[i] = "".join(n) + "\n" + ("-" * sum(len(d_) for d_ in d)) + "\n" + "".join(d)
    return lagrange_polynomials


if __name__ == "__main__":
    plt.style.use("bmh")
    plt.rcParams["font.size"] = 14

    x_coords_interpolation = [1, 2, 3]
    y_coords_interpolation = [3, 1, 4]

    x_smooth = np.linspace(-4, 10, 200)
    polynomial = interpolation_lagrange(x_coords_interpolation, y_coords_interpolation)

    p = get_lagrange_polynomials(x_coords_interpolation, y_coords_interpolation)

    figure, axes = plt.subplots(1, 1, figsize=(16, 9))

    axes.set_facecolor("#fff")
    axes.set_ylabel("y")
    axes.set_xlabel("x")
    axes.set_xlim(0, 4)
    axes.set_ylim(-2, 5)
    mul_y = True

    # p = Polynomial({2: 1, 0: + 1})
    # q = Polynomial({2: 1, 0: - 1})
    # r = Polynomial({2: 1})

    # axes.plot(x_smooth, [xv * xv + 1 for xv in x_smooth], label="P(x) = " + p.__repr__())
    # axes.plot(
    #     x_smooth,
    #     [xv * xv - 1 for xv in x_smooth],
    #     label="Q(x) = " + q.__repr__()
    # )
    # axes.plot(x_smooth, [xv * xv for xv in x_smooth], label="R(x) = " + r.__repr__())

    # axes.scatter(
    #     [-1, 0, 1], [0, 0, 0], zorder=3,
    #     s=100,
    #     facecolor="r", )

    # axes.axhline(y=0, lw=3, c="k")
    axes.set_title("Lagrange Interpolation")
    #
    # for idx in range(3):
    #     y_interpolated = generate_y_coordinates(x_smooth, p[idx] * (y_coords_interpolation[idx] if mul_y else 1), fill_value=np.nan)

    #     axes.plot(
    #         x_smooth,
    #         y_interpolated,
    #         label=f"y{idx} * L{idx}(x)".translate(SUB) + " = " + (p[idx] * (y_coords_interpolation[idx] if mul_y else 1)).__repr__(),
    #     )

    # scatter = plt.scatter(
    #     [xv for xv in x_coords_interpolation[: idx + 1]] * (2 if idx == 2 else 1),
    #     [1 for _ in x_coords_interpolation[: idx + 1]] + ([0 for _ in x_coords_interpolation[: idx + 1]] if idx == 2 else []),
    #     zorder=3,
    #     s=100,
    #     facecolor="r",
    # )

    axes.plot(x_smooth, generate_y_coordinates(x_smooth, polynomial, fill_value=np.nan), c="r", label="P(x)" + " = " + polynomial.__repr__(), lw=2.5)

    scatter = plt.scatter(
        x_coords_interpolation,
        y_coords_interpolation,
        zorder=3,
        s=100,
        facecolor="r",
    )
    axes.legend(fancybox=True, shadow=True, loc=3)
    plt.tight_layout()
    plt.show()
