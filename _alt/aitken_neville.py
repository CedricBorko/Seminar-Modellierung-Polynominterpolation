from utils import has_duplicates


def interpolate_aitken_neville(x_coords: list[float], y_coords: list[float], value: float) -> float:
    if has_duplicates(x_coords):
        raise ValueError("Interpolation requires distinct x coordinates.")

    if len(x_coords) != len(y_coords):
        raise ValueError("X and Y must have the same length.")

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
            numerator = (value - x_coords[i + j]) * polynomials[i] + (x_coords[i] - value) * polynomials[i + 1]
            denominator = x_coords[i] - x_coords[i + j]
            polynomials[i] = numerator / denominator

    return polynomials[0]


if __name__ == "__main__":
    print(interpolate_aitken_neville([1, 2, 3], [1, 4, 9], 12))
