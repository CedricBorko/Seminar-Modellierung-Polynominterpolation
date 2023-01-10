from __future__ import annotations

import numpy as np

from term import Term


class Polynomial:
    def __init__(self, terms: list[Term] = None) -> None:

        self.terms = terms if terms is not None else []
        self.combine_like_terms()

    def __add__(self, other: Polynomial | Term | float | int) -> Polynomial:
        if not isinstance(other, Polynomial | Term | float | int):
            raise ValueError("Only another Polynomial, a Term or a Float / Integer can be added to an existing Polynomial.")

        new_polynomial = Polynomial()

        if isinstance(other, Polynomial):
            new_polynomial.terms = self.terms + other.terms

        if isinstance(other, Term):
            new_polynomial.terms = self.terms + [other]

        if isinstance(other, float | int):
            new_polynomial.terms = self.terms + [Term(other, 0)]

        new_polynomial.combine_like_terms()
        return new_polynomial

    def __sub__(self, other: Polynomial | Term | float | int) -> Polynomial:
        if not isinstance(other, Polynomial | Term | float | int):
            raise ValueError("Only another Polynomial, a Term or a Float / Integer can be subtracted from an existing Polynomial.")

        new_polynomial = Polynomial()

        if isinstance(other, Polynomial):
            new_polynomial.terms = self.terms + [term.inverted() for term in other.terms]

        if isinstance(other, Term):
            new_polynomial.terms = self.terms + [other.inverted()]

        if isinstance(other, float | int):
            new_polynomial.terms = self.terms + [Term(-other, 0)]

        new_polynomial.combine_like_terms()
        return new_polynomial

    def __mul__(self, other: Polynomial | Term | float | int) -> Polynomial:
        if not isinstance(other, Polynomial | Term | float | int):
            raise ValueError("Only another Polynomial, a Term or a Float / Integer can be multiplied to an existing Polynomial.")

        new_polynomial = Polynomial()

        if isinstance(other, Polynomial):
            new_polynomial.terms = [t1 * t2 for t1 in self.terms for t2 in other.terms]

        if isinstance(other, Term | float | int):
            new_polynomial.terms = [term * other for term in self.terms]

        new_polynomial.combine_like_terms()
        return new_polynomial

    def __truediv__(self, other: Term | float | int) -> Polynomial:
        if not isinstance(other, Term | float | int):
            raise ValueError("A Polynomial can only be divided by a Term or a Float / Integer.")

        new_polynomial = Polynomial()
        new_polynomial.terms = [term / other for term in self.terms]
        new_polynomial.combine_like_terms()
        return new_polynomial

    def combine_like_terms(self) -> None:

        self.terms = [
            Term(sum([term.factor for term in self.terms if term.power == power]), power) for power in set(term.power for term in self.terms)
        ]

        self.terms = list(
            filter(lambda term: term.factor != 0 and not np.isclose(term.factor, 0, rtol=1e-05, atol=1e-08, equal_nan=False), self.terms)
        )

    def __repr__(self) -> str:
        return " + ".join(map(str, sorted([term for term in self.terms], key=lambda term: term.power, reverse=True)))

    def value_at(self, x: float) -> float:
        return sum(term.insert(x) for term in self.terms)

    def round(self) -> Polynomial:
        return Polynomial([term.rounded() for term in self.terms])

    def y_intercept(self) -> float:
        return self.value_at(0)


if __name__ == "__main__":

    p1 = Polynomial([Term(1, 4), Term(2, 2)])

    print(p1 )
