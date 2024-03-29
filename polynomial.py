from __future__ import annotations

from dataclasses import dataclass

Number = int | float
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def power_to_superscript(power: int) -> str:
    superscript_power = ""
    for symbol in str(power):
        superscript_power += symbol.translate(SUP)
    return superscript_power


POWER = int
COEFFICIENT = float


@dataclass
class Polynomial:
    def __init__(self, coefficients: dict[POWER, COEFFICIENT] = None) -> None:
        self.coefficients = coefficients if coefficients is not None else {0: 0}
        self.remove_empty_coefficients()

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def powers(self) -> set[int]:
        return set(self.coefficients.keys())

    def __len__(self) -> int:
        return len(self.coefficients)

    def __call__(self, value: float) -> float:
        return sum(factor * value ** power for power, factor in self.coefficients.items())

    def __add__(self, other: Polynomial | Number) -> Polynomial:
        if isinstance(other, Number):
            return Polynomial(
                {power: self[power] + (other if power == 0 else 0) for power in self.powers.union({0})}
            )
        powers = self.powers.union(other.powers)
        return Polynomial({power: self[power] + other[power] for power in powers})

    def __radd__(self, other) -> Polynomial:
        return Polynomial.__add__(self, other)

    def __sub__(self, other: Polynomial | Number) -> Polynomial:
        if isinstance(other, Number):
            return Polynomial(
                {power: self[power] - (other if power == 0 else 0) for power in self.powers.union({0})}
            )

        powers = self.powers.union(other.powers)
        return Polynomial({power: self[power] - other[power] for power in powers})

    def __mul__(self, other: Polynomial | Number) -> Polynomial:
        if isinstance(other, Number):
            return Polynomial({power: self[power] * other for power in self.powers})
        max_power = max(self.powers) + max(other.powers)
        result = [0] * (max_power + 1)

        for left in self.powers:
            for right in other.powers:
                result[left + right] += self[left] * other[right]

        return Polynomial({idx: result[idx] for idx in range(max_power + 1)})

    def __truediv__(self, other: Polynomial | Number) -> Polynomial:
        if isinstance(other, Number):
            return Polynomial({power: self[power] / other for power in self.powers})

        if len(other) != 1:
            raise NotImplementedError()

        dividing_by = other.powers.pop()
        return Polynomial({power - dividing_by: self[power] / other[dividing_by] for power in self.powers})

    def __getitem__(self, power: int) -> float:
        """
        Returns:
            float: The coefficient for a given power or 0 if it doesn't exist.
        """
        return self.coefficients.get(power, 0)

    def __repr__(self) -> str:
        if not self.coefficients:
            return ""

        return ("-" if self[max(self.powers)] < 0 else "") + "".join(
            f"{(' + ' if coefficient >= 0 else ' - ') if idx > 0 else ''}{round(abs(coefficient), 6) if abs(coefficient) != 1 or power == 0 else ''}{('x' if power != 0 else '') + (power_to_superscript(power) if power not in (0, 1) else '')}"
            for idx, (power, coefficient) in
            enumerate(sorted(filter(lambda c: abs(c[1]) > 0.00001, self.coefficients.items()), reverse=True))
        )

    def remove_empty_coefficients(self) -> None:
        """
        Remove all powers with a coefficient of 0. E.g. 0x³ + 2x² + 4x -> 2x² + 4x
        """
        new_coefficients = {power: coefficient for power, coefficient in self.coefficients.items() if
                            coefficient != 0}
        self.coefficients = new_coefficients

    def y_intercept(self) -> float:
        """
        Returns:
            float: Result of the Polynomial equation at x = 0
        """
        return self(0)

    def derivative(self) -> Polynomial:
        return Polynomial({power - 1: self[power] * power for power in self.powers if power != 0})

    @classmethod
    def one(cls: Polynomial) -> Polynomial:
        return Polynomial({0: 1})

    @classmethod
    def x(cls: Polynomial) -> Polynomial:
        return Polynomial({1: 1})

    def __pow__(self, power, modulo = None) -> Polynomial:
        if power < 0: raise ValueError("Cannot raise to negative power (yet).")
        if power == 0: return Polynomial.one()
        if power == 1: return self
        if power == 2: return self * self
        else: return (self * self) ** (power - 1)


if __name__ == "__main__":
    p = Polynomial({3: 4, 2: 1, 0: 5})
    print(p)
    print(p.derivative())
