import re
from abc import ABC
from dataclasses import dataclass
from typing import Set, Optional, Callable, List, Tuple, cast, Dict

import antlr4
import z3
from antlr4 import InputStream
from isla import language
from isla.isla_predicates import STANDARD_STRUCTURAL_PREDICATES, STANDARD_SEMANTIC_PREDICATES
from isla.language import ISLaEmitter, StructuralPredicate, SemanticPredicate, VariableManager, Variable, Formula, \
    parse_tree_text, antlr_get_text_with_whitespace, ISLaUnparser, MExprEmitter, BailPrintErrorStrategy, \
    ensure_unique_bound_variables, VariablesCollector
from isla.type_defs import Grammar
from isla.z3_helpers import get_symbols, smt_expr_to_str
from orderedset import OrderedSet

from islearn.isla_language import IslaLanguageListener
from islearn.islearn_predicates import INTERNET_CHECKSUM_PREDICATE, HEX_TO_DEC_PREDICATE
from islearn.isla_language.IslaLanguageLexer import IslaLanguageLexer
from islearn.isla_language.IslaLanguageParser import IslaLanguageParser
from islearn.mexpr_lexer.MexprLexer import MexprLexer
from islearn.mexpr_parser import MexprParserListener
from islearn.mexpr_parser.MexprParser import MexprParser

NONTERMINAL_PLACEHOLDER = "<?NONTERMINAL>"
MEXPR_PLACEHOLDER = "<?MATCHEXPR>"
STRING_PLACEHOLDER = "<?STRING>"
DSTRINGS_PLACEHOLDER = "<?DSTRINGS>"


class PlaceholderVariable(language.BoundVariable, ABC):
    pass


@dataclass(frozen=True, eq=True, init=True)
class NonterminalPlaceholderVariable(PlaceholderVariable):
    name: str
    n_type: str = NONTERMINAL_PLACEHOLDER


@dataclass(frozen=True, eq=True, init=True)
class NonterminalStringPlaceholderVariable(PlaceholderVariable):
    name: str

    def __str__(self):
        return NONTERMINAL_PLACEHOLDER


@dataclass(frozen=True, eq=True, init=True)
class StringPlaceholderVariable(PlaceholderVariable):
    name: str

    def __str__(self):
        return STRING_PLACEHOLDER


@dataclass(frozen=True, eq=True, init=True)
class DisjunctiveStringsPlaceholderVariable(PlaceholderVariable):
    name: str

    def __str__(self):
        return DSTRINGS_PLACEHOLDER


StringPlaceholderVariableTypes = StringPlaceholderVariable | DisjunctiveStringsPlaceholderVariable


@dataclass(frozen=True, eq=True, init=True)
class MexprPlaceholderVariable(PlaceholderVariable):
    name: str
    variables: Tuple[Variable]
    n_type: str = MEXPR_PLACEHOLDER

    def substitute_variables(self, subst_map: Dict[Variable, Variable]) -> 'MexprPlaceholderVariable':
        return MexprPlaceholderVariable(
            self.name,
            tuple([cast(Variable, subst_map.get(var, var)) for var in self.variables])
        )

    def __str__(self):
        def arg_to_str(v: Variable):
            if isinstance(v, NonterminalPlaceholderVariable):
                return v.name
            assert not isinstance(v, PlaceholderVariable)
            return f"{v.n_type} {v.name}"

        return MEXPR_PLACEHOLDER[:-1] + "(" + ", ".join(map(arg_to_str, self.variables)) + ")>"


class AbstractBindExpression(language.BindExpression):
    def __init__(self, *bound_elements: MexprPlaceholderVariable | str | language.BoundVariable | List[str]):
        super().__init__(*bound_elements)
        assert sum(bool(isinstance(elem, MexprPlaceholderVariable)) for elem in bound_elements) <= 1

        # Below assignment is for type checking reasons only
        self.bound_elements: List[
            MexprPlaceholderVariable |
            language.BoundVariable |
            List[language.BoundVariable]] = self.bound_elements

    def bound_variables(self) -> OrderedSet[language.BoundVariable]:
        # Not isinstance(var, BoundVariable) since we want to exclude dummy variables
        result = OrderedSet([var for var in self.bound_elements if type(var) is language.BoundVariable])
        result.update([
            var for mexpr_ph in self.bound_elements
            if isinstance(mexpr_ph, MexprPlaceholderVariable)
            for var in mexpr_ph.variables
            if isinstance(var, language.BoundVariable)])
        return result

    def substitute_variables(self, subst_map: Dict[Variable, Variable]) -> 'AbstractBindExpression':
        return AbstractBindExpression(
            *[elem if isinstance(elem, list)
              else (
                elem.substitute_variables(subst_map)
                if isinstance(elem, MexprPlaceholderVariable)
                else subst_map.get(elem, elem))
              for elem in self.bound_elements])

    def __str__(self):
        return ''.join(map(
            lambda e: f'{str(e)}'
            if isinstance(e, str)
            else ("[" + "".join(map(str, e)) + "]") if isinstance(e, list)
            else (f"{{{'' if isinstance(e, MexprPlaceholderVariable) else e.n_type + ' '}{str(e)}}}"
                  if not isinstance(e, language.DummyVariable)
                  else (str(e))), self.bound_elements))


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
                    n_type == STRING_PLACEHOLDER or
                    n_type == DSTRINGS_PLACEHOLDER), \
                f"Unknown nonterminal type {n_type} for variable {name}"

        try:
            return next(var for var_name, var in self.variables.items() if var_name == name)
        except StopIteration:
            pass

        if n_type == NONTERMINAL_PLACEHOLDER:
            return self.variables.setdefault(name, NonterminalPlaceholderVariable(name))

        if n_type == STRING_PLACEHOLDER:
            return self.variables.setdefault(name, StringPlaceholderVariable(name))

        if n_type == DSTRINGS_PLACEHOLDER:
            return self.variables.setdefault(name, DisjunctiveStringsPlaceholderVariable(name))

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


def used_variables_in_concrete_syntax(inp: str | IslaLanguageParser.StartContext) -> OrderedSet[str]:
    if isinstance(inp, str):
        lexer = IslaLanguageLexer(InputStream(inp))
        parser = IslaLanguageParser(antlr4.CommonTokenStream(lexer))
        parser._errHandler = BailPrintErrorStrategy()
        context = parser.start()
    else:
        assert isinstance(inp, IslaLanguageParser.StartContext)
        context = inp

    collector = ConcreteSyntaxUsedVariablesCollector()
    antlr4.ParseTreeWalker().walk(collector, context)
    return collector.used_variables


class ConcreteSyntaxUsedVariablesCollector(IslaLanguageListener.IslaLanguageListener):
    def __init__(self):
        self.used_variables: OrderedSet[str] = OrderedSet()

    def collect_used_variables_in_mexpr(self, inp: str) -> None:
        lexer = MexprLexer(InputStream(inp))
        parser = MexprParser(antlr4.CommonTokenStream(lexer))
        parser._errHandler = BailPrintErrorStrategy()
        collector = ConcreteSyntaxMexprUsedVariablesCollector()
        antlr4.ParseTreeWalker().walk(collector, parser.matchExpr())
        self.used_variables.update(collector.used_variables)

    def enterForall(self, ctx: IslaLanguageParser.ForallContext):
        if ctx.varId:
            self.used_variables.add(parse_tree_text(ctx.varId))

    def enterForallMexpr(self, ctx: IslaLanguageParser.ForallMexprContext):
        if ctx.varId:
            self.used_variables.add(parse_tree_text(ctx.varId))
        self.collect_used_variables_in_mexpr(antlr_get_text_with_whitespace(ctx.STRING())[1:-1])

    def enterExists(self, ctx: IslaLanguageParser.ExistsContext):
        if ctx.varId:
            self.used_variables.add(parse_tree_text(ctx.varId))

    def enterExistsMexpr(self, ctx: IslaLanguageParser.ExistsMexprContext):
        if ctx.varId:
            self.used_variables.add(parse_tree_text(ctx.varId))
        self.collect_used_variables_in_mexpr(antlr_get_text_with_whitespace(ctx.STRING())[1:-1])


class ConcreteSyntaxMexprUsedVariablesCollector(MexprParserListener.MexprParserListener):
    def __init__(self):
        self.used_variables: OrderedSet[str] = OrderedSet()

    def enterMatchExprVar(self, ctx: MexprParser.MatchExprVarContext):
        self.used_variables.add(parse_tree_text(ctx.ID()))


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
        self.next_dstrings_placeholder_index = 1

    def enterStart(self, ctx: IslaLanguageParser.StartContext):
        self.used_variables = used_variables_in_concrete_syntax(ctx)

    def exitStart(self, ctx: IslaLanguageParser.StartContext):
        try:
            formula: Formula = self.mgr.create(self.formulas[ctx.formula()])
            formula = ensure_unique_bound_variables(formula)
            self.used_variables.update({var.name for var in VariablesCollector.collect(formula)})
            self.result = \
                self.close_over_xpath_expressions(
                    self.close_over_free_nonterminals(formula))

            # NOTE: We're skipping the free variables check from the standard `ISLaEmitter`
            #       since placeholders and variables in match expression placeholders would
            #       also show up (the latter would be a little tricky to distinguish from
            free_variables = [
                var for var in self.result.free_variables()
                if not isinstance(var, language.Constant)
                   and not isinstance(var, PlaceholderVariable)]
            if free_variables:
                raise SyntaxError('Unbound variables: ' + ', '.join(map(str, free_variables)))
        except RuntimeError as exc:
            raise SyntaxError(str(exc))

    def exitPredicateArg(self, ctx: IslaLanguageParser.PredicateArgContext):
        text = parse_tree_text(ctx)

        if ctx.ID():
            self.predicate_args[ctx] = self.get_var(text)
        elif ctx.INT():
            self.predicate_args[ctx] = int(text)
        elif ctx.STRING():
            self.predicate_args[ctx] = text[1:-1]
        elif ctx.VAR_TYPE():
            variable = self.register_var_for_free_nonterminal(parse_tree_text(ctx.VAR_TYPE()))
            self.predicate_args[ctx] = variable
        elif ctx.XPATHEXPR():
            variable = self.register_var_for_xpath_expression(parse_tree_text(ctx))
            self.predicate_args[ctx] = variable
        elif text == STRING_PLACEHOLDER:
            self.predicate_args[ctx] = StringPlaceholderVariable(
                f"STRING_{self.next_string_placeholder_index}")
            self.next_string_placeholder_index += 1
        elif text == DSTRINGS_PLACEHOLDER:
            self.predicate_args[ctx] = StringPlaceholderVariable(
                f"DSTRINGS_{self.next_dstrings_placeholder_index}")
            self.next_dstrings_placeholder_index += 1
        elif text == NONTERMINAL_PLACEHOLDER:
            self.predicate_args[ctx] = NonterminalStringPlaceholderVariable(
                f"NONTERMINAL_{self.next_nonterminal_string_placeholder_index}")
            self.next_nonterminal_string_placeholder_index += 1
        else:
            assert False, f"Unknown predicate argument type: {text}"

    def exitSMTFormula(self, ctx: IslaLanguageParser.SMTFormulaContext):
        formula_text = self.smt_expressions[ctx.sexpr()]
        formula_text = formula_text.replace(r'\"', '""')

        # We have to replace XPath expressions in the formula, since they can break
        # the parsing (e.g., with `<a>[2]` indexed expressions).
        for xpath_expr in self.vars_for_xpath_expressions:
            formula_text = formula_text.replace(xpath_expr, f'var_{abs(hash(xpath_expr))}')

        match = re.search("(" + re.escape(STRING_PLACEHOLDER) + ")", formula_text)
        if match:
            for group_idx in range(1, 1 + len(match.groups())):
                new_var = self.mgr.bv(self.mgr.fresh_name("STRING"), STRING_PLACEHOLDER)
                fr, to = match.span(group_idx)
                formula_text = formula_text[:fr] + new_var.name + formula_text[to:]

        match = re.search("(" + re.escape(DSTRINGS_PLACEHOLDER) + ")", formula_text)
        if match:
            for group_idx in range(1, 1 + len(match.groups())):
                new_var = self.mgr.bv(self.mgr.fresh_name("DSTRINGS"), DSTRINGS_PLACEHOLDER)
                fr, to = match.span(group_idx)
                formula_text = formula_text[:fr] + new_var.name + formula_text[to:]

        try:
            z3_constr = z3.parse_smt2_string(
                f"(assert {formula_text})",
                decls=({var: z3.String(var)
                        for var in self.known_var_names()} |
                       {nonterminal: z3.String(var.name)
                        for nonterminal, var in self.vars_for_free_nonterminals.items()} |
                       {f'var_{abs(hash(xpath_expr))}': z3.String(var.name)
                        for xpath_expr, var in self.vars_for_xpath_expressions.items()}))[0]
        except z3.Z3Exception as exp:
            raise SyntaxError(
                f"Error parsing SMT formula '{formula_text}', {exp.value.decode().strip()}")

        free_vars = [self.get_var(str(s)) for s in get_symbols(z3_constr)]
        self.formulas[ctx] = language.SMTFormula(z3_constr, *free_vars)

    def exitSexprNonterminalStringPh(self, ctx: IslaLanguageParser.SexprNonterminalStringPhContext):
        self.smt_expressions[ctx] = parse_tree_text(ctx)

    def exitSexprStringPh(self, ctx: IslaLanguageParser.SexprStringPhContext):
        self.smt_expressions[ctx] = parse_tree_text(ctx)

    def exitSexprDisjStringsPh(self, ctx: IslaLanguageParser.SexprDisjStringsPhContext):
        self.smt_expressions[ctx] = parse_tree_text(ctx)

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
        return AbstractBindExpression(*mexpr_emitter.result)


class AbstractMExprEmitter(MExprEmitter, MexprParserListener.MexprParserListener):
    def __init__(self, mgr: AbstractVariableManager):
        super().__init__(mgr)
        self.mgr = mgr

    def exitMatchExprPlaceholder(self, ctx: MexprParser.MatchExprPlaceholderContext):
        param_defs: List[MexprParser.MexprPlaceholderParamContext] = ctx.mexprPlaceholderParam()
        params: List[language.Variable] = []
        for param_def in param_defs:
            param_id = parse_tree_text(param_def.ID())
            if param_def.varType() is not None:
                params.append(self.mgr.bv(param_id, parse_tree_text(param_def.varType())))
            else:
                params.append(self.mgr.bv(param_id, NONTERMINAL_PLACEHOLDER))

        name = self.mgr.fresh_name("mexprPlaceholder")
        mexpr_placeholder = MexprPlaceholderVariable(name, tuple(params))
        self.mgr.variables.setdefault(name, mexpr_placeholder)
        self.result.append(mexpr_placeholder)


ISLEARN_STANDARD_SEMANTIC_PREDICATES = STANDARD_SEMANTIC_PREDICATES | {
    INTERNET_CHECKSUM_PREDICATE,
    HEX_TO_DEC_PREDICATE,
}


def parse_abstract_isla(
        inp: str,
        grammar: Optional[Grammar] = None,
        structural_predicates: Set[StructuralPredicate] = STANDARD_STRUCTURAL_PREDICATES,
        semantic_predicates: Set[SemanticPredicate] = ISLEARN_STANDARD_SEMANTIC_PREDICATES) -> Formula:
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

    def _unparse_smt_formula(self, formula: language.SMTFormula):
        result = smt_expr_to_str(formula.formula)

        for variable in formula.free_variables():
            if isinstance(variable, StringPlaceholderVariableTypes):
                result = result.replace(variable.name, str(variable))

        return [result]


def unparse_abstract_isla(formula: Formula, indent="  "):
    return AbstractISLaUnparser(formula, indent).unparse()
