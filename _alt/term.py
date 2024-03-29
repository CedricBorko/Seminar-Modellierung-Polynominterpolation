from __future__ import annotations

from utils import SUP


class Term:
    def __init__(self, factor: float, power: int) -> None:
        self.factor = factor
        self.power = power

    def __repr__(self) -> str:
        return f"{'-' if self.factor == -1 else '' if self.factor == 1 and self.power != 0 else self.factor}{'x' + str(self.power).translate(SUP) if self.power != 0 else ''}"

    def __sub__(self, other: "Polynomial" | Term | float | int) -> Term | "Polynomial":
        from polynomial import Polynomial

        if isinstance(other, Polynomial):
            return Polynomial([term.inverted() for term in other.terms] + [self])

        if isinstance(other, Term):
            return Term(self.factor - other.factor, self.power) if other.power == self.power else Polynomial([self, other.inverted()])

        if isinstance(other, float | int):
            return Polynomial([self, Term(other, 0)])

    def __mul__(self, other: Term | float | int) -> Term:
        if isinstance(other, Term):
            return Term(self.factor * other.factor, self.power + other.power)
        if isinstance(other, float | int):
            return Term(self.factor * other, self.power)

    def __truediv__(self, other: Term | float | int) -> Term:
        if isinstance(other, Term) and other.factor != 0:
            return Term(self.factor / other.factor, self.power - other.power)

        if isinstance(other, float | int) and other != 0:
            return Term(self.factor / other, self.power)

    def differentiate(self) -> Term:
        return Term(self.factor * self.power, self.power - 1) if self.power != 0 else Term(0, 0)

    def inverted(self) -> Term:
        return Term(-self.factor, self.power)

    def insert(self, value: float) -> float:
        return self.factor * value**self.power

    def rounded(self) -> Term:
        return Term(round(self.factor, 6), self.power)
