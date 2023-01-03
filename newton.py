from functools import reduce
from typing import Collection

from polynomial import Polynomial
from term import Term
from utils import Point, has_duplicates


def interpolate_newton(points: list[Point]) -> float:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    degree = len(points) - 1
    # f0 = P(x0) = a0
    # f1 = P(x1) = a0 + a1(x1 - x0)
    # P(x) = a0 +a1(x - x0) + a2(x - x0)(x - x1) + ... + an(x - x0) ... (x - xn)

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
    print(interpolate_newton([Point(1, 2), Point(2, 3), Point(3, 1), Point(4, 3)]))
