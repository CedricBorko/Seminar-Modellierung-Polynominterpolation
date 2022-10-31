from __future__ import annotations

from dataclasses import dataclass
from decimal import DivisionByZero
from typing import Collection

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


@dataclass
class Point:

    x: float
    y: float


@dataclass
class Term:

    value: float
    power: float

    def __add__(self, other: Term | float) -> Term | Polynomial:
        if not isinstance(other, Term):
            raise ValueError(f"A Term can only be added to another Term. Got {type(other)}")

        if other.power == self.power:
            return Term(self.value + other.value, self.power)

        raise ValueError(f"Terms of different powers cannot be added together.")

    def __sub__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            raise ValueError(f"A Term can only be subtracted from another Term. Got {type(other)}")

        if other.power == self.power:
            return Term(self.value - other.value, self.power)

        raise ValueError(f"Terms of different powers cannot be subtracted.")

    def __mul__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            raise ValueError(f"A Term can only be multiplied to another Term. Got {type(other)}")

        return Term(self.value * other.value, self.power + other.power)

        
    def __truediv__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            raise ValueError(f"A Term can only be divided by another Term. Got {type(other)}")

        if other.value == 0:
            raise DivisionByZero(f"Cannot divide by zero.")

        return Term(self.value / other.value, self.power - other.power)


    def __repr__(self) -> str:
        repr_string = "-" if self.value == -1 else str(self.value) if self.value != 1 else "x"
        repr_string += "x" if self.power != 0 and "x" not in repr_string else ""
        repr_string += str(self.power).translate(SUP) if self.power not in (0, 1) else ""

        return repr_string

    def derivative(self) -> Term | None:
        if self.power == 0: return None
        return Term(self.value * self.power, self.power - 1)

class Polynomial:
    
    def __init__(self, terms: list[Term] = None) -> None:
        self.terms = terms if terms is not None else []
        self.combine_like_terms()
        

    @classmethod
    def from_terms(cls: Polynomial, terms: Collection[Term]) -> Polynomial:
        return Polynomial(terms)

    def __add__(self, other: Polynomial | Term) -> Polynomial:
        
        new_polynomial = Polynomial(self.terms)

        if isinstance(other, Polynomial):
            new_polynomial.terms.extend(other.terms)
        
        elif isinstance(other, Term):
            new_polynomial.terms.append(other)

        else:
            raise ValueError("Can only add Terms or other Polynomials to an existing one.")

        new_polynomial.combine_like_terms()
        return new_polynomial

    def __mul__(self, other: Polynomial | Term) -> Polynomial:
        
        new_polynomial = Polynomial()

        if isinstance(other, Polynomial):
            new_polynomial.terms = [t1 * t2 for t1 in self.terms for t2 in other.terms]
        
        elif isinstance(other, Term):
            new_polynomial.terms = [term * other for term in self.terms]

        else:
            raise ValueError("Can only add Terms or other Polynomials to an existing one.")

        new_polynomial.combine_like_terms()
        return new_polynomial
        
    def combine_like_terms(self) -> None:
        powers = set(term.power for term in self.terms)
        term_dict = {
            power: [term.value for term in self.terms if term.power == power] for power in powers
        }

        self.terms = [Term(sum(term_dict[power]), power) for power in term_dict]

    @property
    def degree(self) -> int:
        return max(term.power for term in self.terms)


    def __repr__(self) -> str:
        return ' + '.join(map(str, sorted([term for term in self.terms], key=lambda term: term.power, reverse=True)))
    
    def derivative(self) -> Polynomial:
        return Polynomial(list(filter(
            lambda term: term is not None, [term.derivative() for term in self.terms]
        )))

def has_duplicates(points: list[Point]) -> bool:
    return len(p.x for p in points) == len(set(p.x for p in points))



