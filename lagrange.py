from polynomial import Polynomial
from term import Term
from utils import Point, has_duplicates

# Degree <= k with Datapoints l0, l1, ..., lk of degree k

#       (X - X0)           (X - Xj-1)      (X - Xj+1)      (X - Xk)
# Lj = ---------- * ... * ------------- * ------------- * -------------
#       (Xj - X0)          (Xj - Xj-1)     (Xj - Xj+1)     (Xj - Xk)

# Product over 0 <= m <= k; m != j; (x - xm) / (xj - xm)


def interpolate_lagrange(points: list[Point]) -> Polynomial:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    degree = len(points) - 1

    results = []

    for j in range(0, degree + 1):

        start = Polynomial([Term(1, 0)])

        for m in range(0, degree + 1):
            if m == j:
                continue

            top = Polynomial([Term(1, 1), Term(-points[m].x, 0)])
            bottom = points[j].x - points[m].x
            start *= top / bottom

        results.append(start * points[j].y)

    sum_poly = Polynomial()
    for poly in results:
        sum_poly += poly
    return sum_poly.round()
