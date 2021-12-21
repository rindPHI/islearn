from enum import Enum
from typing import Callable, Iterable, Union

from isla import isla


class Placeholder:
    class PlaceholderType(Enum):
        NONTERMINAL = 0
        NUMBER = 1

    def __init__(self, name: str, ph_type: PlaceholderType):
        self.name = name
        self.ph_type = ph_type


class Pattern:
    def __init__(
            self,
            placeholders: Iterable[Placeholder],
            precondition: Callable[[Union[int, str], ...], bool],
            formula_factory: Callable[[Union[int, str], ...], isla.Formula]):
        self.placeholders = placeholders
        self.precondition = precondition
        self.formula_factory = formula_factory

    def __eq__(self, other) -> bool:
        return (isinstance(other, Pattern) and
                (self.placeholders, self.precondition, self.formula_factory) ==
                (other.placeholders, other.precondition, other.formula_factory))

    def __repr__(self) -> str:
        return f"Pattern(" \
               f"{repr(self.placeholders)}, " \
               f"{self.precondition.__name__}, " \
               f"{self.formula_factory.__name__})"

    def __hash__(self) -> int:
        return hash((self.placeholders, self.precondition, self.formula_factory))
