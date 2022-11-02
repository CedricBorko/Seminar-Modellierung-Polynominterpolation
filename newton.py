from functools import reduce

from utils import Point, has_duplicates


def newton_interpolation(points: list[Point]) -> float:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    degree = len(points) - 1
    # f0 = P(x0) = a0
    # f1 = P(x1) = a0 + a1(x1 - x0)
    # P(x) = a0 +a1(x - x0) + a2(x - x0)(x - x1) + ... + an(x - x0) ... (x - xn)

    parameters = [points[0].y]

    # f0 = a0
    # f1 = a0 + a1(x1 - x0) <=> a1 = (f1 - a0) / (x1 - x0)
    # f2 = a0 + a1(x1 - x0) + a2(x2 -x0)(x2 - x1) <=> a2 = (f2 - a0 - a1(x1 - x0)) / ((x2 - x0)(x2 - x1))

    # for n in range(1, degree + 1):
    #     fn = points[n].y
    #     an = fn - parameters[0]
    #     factors = [(p1.x, p2.x) for p1, p2 in zip([points[n]] * (n), points[:n])]

    #     prod = parameters[n] if n < len(parameters) else 1
    #     for r in range(n):
    #         used_factors = factors[: r + 1]
    #         l, r = used_factors[r]
    #         prod *= l - r

    #     if n >= len(parameters):
    #         an /= prod
    #     an -= prod
    #     parameters.append(an)
    # return parameters


if __name__ == "__main__":
    print(newton_interpolation([Point(0, 1), Point(1, 3), Point(2, 2)]))
