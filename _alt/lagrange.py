from polynomial_ import Polynomial
from utils import has_duplicates

# Degree <= k with Datapoints l0, l1, ..., lk of degree k

#       (X - X0)           (X - Xj-1)      (X - Xj+1)      (X - Xk)
# Lj = ---------- * ... * ------------- * ------------- * -------------
#       (Xj - X0)          (Xj - Xj-1)     (Xj - Xj+1)     (Xj - Xk)

# Product over 0 <= m <= k; m != j; (x - xm) / (xj - xm)


def interpolate_lagrange(x_coords: list[float], y_coords: list[float]) -> Polynomial:
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
    return sum_poly


if __name__ == "__main__":
    p = interpolate_lagrange([1, 2, 3], [1, 4, 9])
    print(p(12))
