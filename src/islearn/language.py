from abc import ABC
from dataclasses import dataclass
from typing import Set, Optional, Callable

import antlr4
from antlr4 import InputStream
from isla import language
from isla.isla_predicates import STANDARD_STRUCTURAL_PREDICATES, STANDARD_SEMANTIC_PREDICATES
from isla.language import ISLaEmitter, StructuralPredicate, SemanticPredicate, VariableManager, Variable, Formula, \
    parse_tree_text
from isla.type_defs import Grammar

from islearn.isla_language.IslaLanguageLexer import IslaLanguageLexer
from islearn.isla_language.IslaLanguageParser import IslaLanguageParser

NONTERMINAL_PLACEHOLDER = "<?NONTERMINAL>"


class PlaceholderVariable(language.BoundVariable, ABC):
    pass


@dataclass(frozen=True, init=True)
class NonterminalPlaceholderVariable(PlaceholderVariable):
    name: str
    n_type: str = NONTERMINAL_PLACEHOLDER


@dataclass(frozen=True, init=True)
class NonterminalStringPlaceholderVariable(PlaceholderVariable):
    name: str

    def __str__(self):
        return NONTERMINAL_PLACEHOLDER


class AbstractVariableManager(VariableManager):
    def __init__(self, grammar: Grammar):
        super().__init__(grammar)

    def _var(self,
             name: str,
             n_type: Optional[str],
             constr: Optional[Callable[[str, Optional[str]], Variable]] = None) -> Variable:
        if n_type is not None:
            assert n_type == Variable.NUMERIC_NTYPE or n_type in self.grammar or n_type == NONTERMINAL_PLACEHOLDER, \
                f"Unknown nonterminal type {n_type} for variable {name}"

        try:
            return next(var for var_name, var in self.variables.items() if var_name == name)
        except StopIteration:
            pass

        if n_type == NONTERMINAL_PLACEHOLDER:
            return self.variables.setdefault(name, NonterminalPlaceholderVariable(name))

        if constr is not None and n_type:
            return self.variables.setdefault(name, constr(name, n_type))

        matching_placeholders = [var for var_name, var in self.placeholders.items() if var_name == name]
        if matching_placeholders:
            return matching_placeholders[0]

        assert constr is not None
        return self.placeholders.setdefault(name, constr(name, None))


class AbstractISLaEmitter(ISLaEmitter):
    def __init__(
            self,
            grammar: Grammar,
            structural_predicates: Set[StructuralPredicate] = STANDARD_STRUCTURAL_PREDICATES,
            semantic_predicates: Set[SemanticPredicate] = STANDARD_SEMANTIC_PREDICATES):
        super().__init__(grammar, structural_predicates, semantic_predicates)

        self.mgr = AbstractVariableManager(grammar)
        self.next_nonterminal_string_placeholder_index = 1

    def exitPredicateArg(self, ctx: IslaLanguageParser.PredicateArgContext):
        text = parse_tree_text(ctx)

        if ctx.ID():
            self.predicate_args[ctx] = self.get_var(text)
        elif ctx.INT():
            self.predicate_args[ctx] = int(text)
        elif ctx.STRING():
            self.predicate_args[ctx] = text[1:-1]
        elif text == NONTERMINAL_PLACEHOLDER:
            self.predicate_args[ctx] = NonterminalStringPlaceholderVariable(
                f"NONTERMINAL_{self.next_nonterminal_string_placeholder_index}")
            self.next_nonterminal_string_placeholder_index += 1
        else:
            assert False, f"Unknown predicate argument type: {text}"


def parse_abstract_isla(
        inp: str,
        grammar: Grammar,
        structural_predicates: Set[StructuralPredicate] = STANDARD_STRUCTURAL_PREDICATES,
        semantic_predicates: Set[SemanticPredicate] = STANDARD_SEMANTIC_PREDICATES) -> Formula:
    class BailPrintErrorStrategy(antlr4.BailErrorStrategy):
        def recover(self, recognizer: antlr4.Parser, e: antlr4.RecognitionException):
            recognizer._errHandler.reportError(recognizer, e)
            super().recover(recognizer, e)

    lexer = IslaLanguageLexer(InputStream(inp))
    parser = IslaLanguageParser(antlr4.CommonTokenStream(lexer))
    parser._errHandler = BailPrintErrorStrategy()
    isla_emitter = AbstractISLaEmitter(grammar, structural_predicates, semantic_predicates)
    antlr4.ParseTreeWalker().walk(isla_emitter, parser.start())
    return isla_emitter.result
