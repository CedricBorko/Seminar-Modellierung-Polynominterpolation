from dataclasses import dataclass
from typing import Collection

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


@dataclass
class Point:

    x: float
    y: float


def has_duplicates(x_coordinates: Collection[float]) -> bool:
    return len(list(x_coordinates)) == len(set(x_coordinates))
