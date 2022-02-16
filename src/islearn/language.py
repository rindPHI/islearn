import re
from abc import ABC
from dataclasses import dataclass
from typing import Set, Optional, Callable, List, Tuple, cast

import antlr4
import z3
from antlr4 import InputStream
from isla import language
from isla.helpers import get_symbols
from isla.isla_predicates import STANDARD_STRUCTURAL_PREDICATES, STANDARD_SEMANTIC_PREDICATES
from isla.language import ISLaEmitter, StructuralPredicate, SemanticPredicate, VariableManager, Variable, Formula, \
    parse_tree_text, antlr_get_text_with_whitespace, ISLaUnparser, MExprEmitter
from isla.type_defs import Grammar

from islearn.isla_language.IslaLanguageLexer import IslaLanguageLexer
from islearn.isla_language.IslaLanguageParser import IslaLanguageParser
from islearn.mexpr_lexer.MexprLexer import MexprLexer
from islearn.mexpr_parser import MexprParserListener
from islearn.mexpr_parser.MexprParser import MexprParser

NONTERMINAL_PLACEHOLDER = "<?NONTERMINAL>"
STRING_PLACEHOLDER = "<?STRING>"
MEXPR_PLACEHOLDER = "<?MATCHEXPR>"


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


@dataclass(frozen=True, init=True)
class StringPlaceholderVariable(PlaceholderVariable):
    name: str

    def __str__(self):
        return STRING_PLACEHOLDER


@dataclass(frozen=True, init=True)
class MexprPlaceholderVariable(PlaceholderVariable):
    name: str
    variables: Tuple[NonterminalPlaceholderVariable]
    n_type: str = MEXPR_PLACEHOLDER

    def __str__(self):
        return MEXPR_PLACEHOLDER[:-1] + "(" + ", ".join(map(str, self.variables)) + ")>"


class AbstractVariableManager(VariableManager):
    def __init__(self, grammar: Optional[Grammar] = None):
        super().__init__(grammar)

    def _var(self,
             name: str,
             n_type: Optional[str],
             constr: Optional[Callable[[str, Optional[str]], Variable]] = None) -> Variable:
        if n_type is not None:
            assert (self.grammar is None or
                    n_type == Variable.NUMERIC_NTYPE or
                    n_type in self.grammar or
                    n_type == NONTERMINAL_PLACEHOLDER or
                    n_type == STRING_PLACEHOLDER), \
                f"Unknown nonterminal type {n_type} for variable {name}"

        try:
            return next(var for var_name, var in self.variables.items() if var_name == name)
        except StopIteration:
            pass

        if n_type == NONTERMINAL_PLACEHOLDER:
            return self.variables.setdefault(name, NonterminalPlaceholderVariable(name))

        if n_type == STRING_PLACEHOLDER:
            return self.variables.setdefault(name, StringPlaceholderVariable(name))

        if constr is not None and n_type:
            return self.variables.setdefault(name, constr(name, n_type))

        matching_placeholders = [var for var_name, var in self.placeholders.items() if var_name == name]
        if matching_placeholders:
            return matching_placeholders[0]

        assert constr is not None
        return self.placeholders.setdefault(name, constr(name, None))

    def fresh_name(self, prefix: str = "v") -> str:
        idx = 0
        while f"{prefix}_{idx}" in self.placeholders or f"{prefix}_{idx}" in self.variables:
            idx += 1

        return f"{prefix}_{idx}"


class AbstractISLaEmitter(ISLaEmitter):
    def __init__(
            self,
            grammar: Optional[Grammar] = None,
            structural_predicates: Set[StructuralPredicate] = STANDARD_STRUCTURAL_PREDICATES,
            semantic_predicates: Set[SemanticPredicate] = STANDARD_SEMANTIC_PREDICATES):
        super().__init__(grammar, structural_predicates, semantic_predicates)

        self.mgr = AbstractVariableManager(grammar)
        self.next_nonterminal_string_placeholder_index = 1
        self.next_string_placeholder_index = 1

    def exitPredicateArg(self, ctx: IslaLanguageParser.PredicateArgContext):
        text = parse_tree_text(ctx)

        if ctx.ID():
            self.predicate_args[ctx] = self.get_var(text)
        elif ctx.INT():
            self.predicate_args[ctx] = int(text)
        elif ctx.STRING():
            self.predicate_args[ctx] = text[1:-1]
        elif text == STRING_PLACEHOLDER:
            self.predicate_args[ctx] = NonterminalStringPlaceholderVariable(
                f"STRING_{self.next_string_placeholder_index}")
            self.next_string_placeholder_index += 1
        elif text == NONTERMINAL_PLACEHOLDER:
            self.predicate_args[ctx] = NonterminalStringPlaceholderVariable(
                f"NONTERMINAL_{self.next_nonterminal_string_placeholder_index}")
            self.next_nonterminal_string_placeholder_index += 1
        else:
            assert False, f"Unknown predicate argument type: {text}"

    def exitSMTFormula(self, ctx: IslaLanguageParser.SMTFormulaContext):
        formula_text = antlr_get_text_with_whitespace(ctx)

        match = re.search("(" + re.escape(STRING_PLACEHOLDER) + ")", formula_text)
        if match:
            for group_idx in range(1, 1 + len(match.groups())):
                new_var = self.mgr.bv(self.mgr.fresh_name("STRING"), STRING_PLACEHOLDER)
                fr, to = match.span(group_idx)
                formula_text = formula_text[:fr] + new_var.name + formula_text[to:]

        try:
            z3_constr = z3.parse_smt2_string(
                f"(assert {formula_text})",
                decls={var: z3.String(var) for var in self.known_var_names()})[0]
        except z3.Z3Exception as exp:
            raise SyntaxError(
                f"Error parsing SMT formula '{formula_text}', {exp.value.decode().strip()}")

        free_vars = [self.get_var(str(s)) for s in get_symbols(z3_constr)]
        self.formulas[ctx] = language.SMTFormula(z3_constr, *free_vars)

    def parse_mexpr(self, inp: str, mgr: VariableManager) -> language.BindExpression:
        class BailPrintErrorStrategy(antlr4.BailErrorStrategy):
            def recover(self, recognizer: antlr4.Parser, e: antlr4.RecognitionException):
                recognizer._errHandler.reportError(recognizer, e)
                super().recover(recognizer, e)

        lexer = MexprLexer(InputStream(inp))
        parser = MexprParser(antlr4.CommonTokenStream(lexer))
        parser._errHandler = BailPrintErrorStrategy()
        mexpr_emitter = AbstractMExprEmitter(mgr)
        antlr4.ParseTreeWalker().walk(mexpr_emitter, parser.matchExpr())
        return language.BindExpression(*mexpr_emitter.result)


class AbstractMExprEmitter(MExprEmitter, MexprParserListener.MexprParserListener):
    def __init__(self, mgr: AbstractVariableManager):
        super().__init__(mgr)
        self.mgr = mgr

    def exitMatchExprPlaceholder(self, ctx: MexprParser.MatchExprPlaceholderContext):
        ids = [parse_tree_text(id) for id in ctx.ID()]
        phs = [cast(NonterminalPlaceholderVariable, self.mgr.bv(id, NONTERMINAL_PLACEHOLDER)) for id in ids]
        name = self.mgr.fresh_name("mexprPlaceholder")
        mexpr_placeholder = MexprPlaceholderVariable(name, tuple(phs))
        self.mgr.variables.setdefault(name, mexpr_placeholder)
        self.result.append(mexpr_placeholder)


def parse_abstract_isla(
        inp: str,
        grammar: Optional[Grammar] = None,
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


class AbstractISLaUnparser(ISLaUnparser):
    def __init__(self, formula: Formula, indent="  "):
        super().__init__(formula, indent)

    def _unparse_match_expr(self, match_expr: language.BindExpression | None) -> str:
        if match_expr is None:
            return ""

        result = ''.join(map(
            lambda e: f'{str(e)}'
            if isinstance(e, str)
            else ("[" + "".join(map(str, e)) + "]") if isinstance(e, list)
            else (f"{{{'' if isinstance(e, MexprPlaceholderVariable) else e.n_type + ' '}{str(e)}}}"
                  if not isinstance(e, language.DummyVariable)
                  else (str(e))), match_expr.bound_elements))

        return f'="{result}"'

    def _unparse_smt_formula(self, formula: language.SMTFormula):
        result = formula.formula.sexpr()

        for variable in formula.free_variables():
            if not isinstance(variable, StringPlaceholderVariable):
                continue

            result = result.replace(variable.name, str(variable))

        return [result]
