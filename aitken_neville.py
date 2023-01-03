from utils import Point, has_duplicates


def interpolate_aitken_neville(points: list[Point], x: float) -> float:
    if has_duplicates(p.x for p in points):
        raise ValueError("Interpolation requires distinct x coordinates.")

    degree = len(points) - 1
    polynomials = {(i, i): points[i].y for i in range(degree + 1)}

    for l in range(degree + 1):
        i = 0
        j = l + 1
        for _ in range(degree - l):

            p_left = polynomials[i + 1, j]
            p_right = polynomials[i, j - 1]

            p_ij = (x - points[i].x) * p_left - (x - points[j].x) * p_right
            p_ij /= points[j].x - points[i].x
            polynomials[i, j] = p_ij

            i += 1
            j += 1

    return polynomials[0, degree]


if __name__ == "__main__":
    print(interpolate_aitken_neville([Point(16, 0.25), Point(64, 0.125), Point(100, 0.1)], 81))
