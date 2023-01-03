from utils import Point, has_duplicates


def interpolate_aitken_neville(points: list[Point], x: float) -> float:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    n = len(points)
    # polynomials = {(i, i): points[i].y for i in range(n)}

    polys = [0] * n

    for l in range(n):
        for i in range(n - l):
            if l == 0:
                polys[i] = points[i].y
            else:
                numerator = (x - points[i + l].x) * polys[i] + (points[i].x - x) * polys[i + 1]
                denominator = points[i].x - points[i + l].x
                polys[i] = numerator / denominator

        # i = 0
        # j = l + 1
        # for _ in range(n):

        #     p_left = polynomials[i + 1, j]
        #     p_right = polynomials[i, j - 1]

        #     p_ij = (x - points[i].x) * p_left - (x - points[j].x) * p_right
        #     p_ij /= points[j].x - points[i].x
        #     polynomials[i, j] = p_ij

        #     i += 1
        #     j += 1

    return polys[0]


if __name__ == "__main__":
    print(interpolate_aitken_neville([Point(1, 0.25), Point(2, 0.5), Point(4, 1)], 8))
