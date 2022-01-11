from enum import Enum
from typing import Callable, Iterable, Union, Any, Dict, Optional

from grammar_graph import gg
from isla import isla
from isla.type_defs import Grammar


class Placeholder:
    class PlaceholderType(Enum):
        NONTERMINAL = 0
        NUMBER = 1

    def __init__(self, name: str, ph_type: PlaceholderType):
        self.name = name
        self.ph_type = ph_type

    def __eq__(self, other):
        return isinstance(other, Placeholder) and (self.name, self.ph_type) == (other.name, other.ph_type)

    def __hash__(self):
        return hash((self.name, self.ph_type))

    def __repr__(self):
        return f"Placeholder({repr(self.name)}, {repr(self.ph_type)})"

    def __str__(self) -> str:
        return self.name


class Placeholders:
    def __init__(self, *placeholders: Placeholder, instantiations: Optional[Dict[Placeholder, Any]] = None):
        self.placeholders = list(placeholders)
        self.instantiations: Dict[Placeholder, Any] = instantiations or dict.fromkeys(placeholders)

    def __repr__(self) -> str:
        return f"Placeholders({', '.join(map(repr, self.placeholders))}, {repr(self.instantiations)})"

    def __str__(self) -> str:
        return "{" + ", ".join([f"{ph} -> {inst}" for ph, inst in self.instantiations.items()]) + "}"

    def __setitem__(self, key: Placeholder, value: Any):
        assert key in self.instantiations
        self.instantiations[key] = value

    def __getitem__(self, idx: Placeholder | int | str) -> Optional[Any]:
        key: Optional[Placeholder] = None

        if isinstance(idx, Placeholder):
            key = idx
        elif isinstance(idx, int):
            key = self.placeholders[idx]
        else:
            key = next(placeholder for placeholder in self.placeholders
                       if placeholder.name == idx)

        assert key is not None

        return self.instantiations.get(key, None)

    def __eq__(self, other):
        return (isinstance(other, Placeholders) and
                (self.placeholders, self.instantiations) == (other.placeholders, other.instantiations))

    def __len__(self):
        return len(self.placeholders)

    def next_uninstantiated_placeholder(self) -> Optional[Placeholder]:
        try:
            return next(placeholder for placeholder in self.placeholders
                        if self.instantiations[placeholder] is None)
        except StopIteration:
            return None

    def instantiate(self, key: Placeholder, value: Any) -> 'Placeholders':
        assert key is not None
        assert key in self.instantiations
        assert value is not None
        return Placeholders(*self.placeholders, instantiations=self.instantiations | {key: value})

    def complete(self):
        return len(self.instantiations) == len(self.placeholders)


class Pattern:
    def __init__(
            self,
            placeholders: Placeholders,
            precondition: Callable[[Placeholders, Grammar, gg.GrammarGraph], Optional[bool]],
            formula_factory: Callable[[Placeholders, Grammar], isla.Formula]):
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
