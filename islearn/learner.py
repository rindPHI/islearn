import itertools
import logging
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional, cast

import z3
from grammar_graph import gg
from isla import isla
from isla.helpers import is_z3_var, is_nonterminal
from isla.isla_predicates import is_before, BEFORE_PREDICATE, COUNT_PREDICATE, reachable
from isla.type_defs import Grammar
from swiplserver import PrologMQI

logger = logging.getLogger("learner")


class PlaceholderVariable(isla.BoundVariable, ABC):
    pass


@dataclass(frozen=True, init=True)
class NonterminalPlaceholderVariable(PlaceholderVariable):
    name: str


@dataclass(frozen=True, init=True)
class NonterminalStringPlaceholderVariable(PlaceholderVariable):
    name: str


def filter_invariants(
        patterns: List[isla.Formula],
        inputs: List[isla.DerivationTree],
        grammar: Grammar) -> List[isla.Formula]:
    candidates: Set[isla.Formula] = set()
    for pattern in patterns:
        var_map: Dict[PlaceholderVariable, str] = {
            placeholder: placeholder.name.upper()
            for placeholder in get_placeholders(pattern)}

        core = get_quantifier_free_core(pattern)
        assert core is not None, "Only supporting formulas with one quantifier chain."

        conjunctive_formulas = split_cnf(core)
        assert conjunctive_formulas is not None, "Only supporting conjunctive quantifier-free core."

        translators: Set[PrologTranslator] = {get_translator(formula, grammar) for formula in conjunctive_formulas}
        translators = {translator for translator in translators if translator is not None}

        for inp in inputs:
            assumptions: List[str] = []

            assumptions.extend([f"inner_node({tree.id})" for _, tree in inp.paths() if tree.num_children() > 0])
            assumptions.extend(get_in_assumptions(inp))
            for translator in translators:
                assumptions.extend(translator.facts(inp))

            query = [f"inner_node({var})" for ph, var in var_map.items()
                     if isinstance(ph, NonterminalPlaceholderVariable)]
            query.extend(get_in_query(pattern, inp.id, var_map))

            query.extend([
                translator.query(conjunct, var_map)
                for conjunct in conjunctive_formulas
                for translator in translators
                if translator.responsible(conjunct)])

            if any(isinstance(var, NonterminalStringPlaceholderVariable) for var in var_map):
                assumptions.extend([f"nonterminal(\"{nonterminal}\")" for nonterminal in grammar])
                query.extend([f"nonterminal({var_map[var]})"
                              for var in var_map
                              if isinstance(var, NonterminalStringPlaceholderVariable)])

            result = evaluate_prolog_query(assumptions, ", ".join(query))
            if result is None:
                continue

            for instantiation in result:
                candidates.add(pattern.substitute_variables({
                    placeholder:
                        isla.BoundVariable(
                            placeholder.name,
                            inp.get_subtree(inp.find_node(instantiation[variable])).value)
                    for placeholder, variable in var_map.items()
                    if isinstance(placeholder, NonterminalPlaceholderVariable)
                }).substitute_expressions({
                    placeholder: instantiation[variable]
                    for placeholder, variable in var_map.items()
                    if isinstance(placeholder, NonterminalStringPlaceholderVariable)
                }))

    return [
        candidate for candidate in candidates
        if all(isla.evaluate(candidate, inp, grammar) for inp in inputs)
    ]


def get_in_query(
        formula: isla.Formula,
        root_id: int,
        var_map: Dict[PlaceholderVariable, str]) -> List[str]:
    query: List[str] = []
    constant = extract_top_level_constant(formula)

    class InVisitor(isla.FormulaVisitor):
        def visit_exists_formula(self, formula: isla.ExistsFormula):
            self.handle(formula)

        def visit_forall_formula(self, formula: isla.ForallFormula):
            self.handle(formula)

        def handle(self, formula: isla.QuantifiedFormula):
            nonlocal query
            bv = cast(NonterminalPlaceholderVariable, formula.bound_variable)
            assert isinstance(bv, NonterminalPlaceholderVariable)
            if formula.in_variable == constant:
                iv = root_id
            else:
                assert isinstance(formula.in_variable, NonterminalPlaceholderVariable)
                iv = var_map[formula.in_variable]

            query.append(f"tin({var_map[bv]}, {iv})")

    formula.accept(InVisitor())

    return query


def get_in_assumptions(tree: isla.DerivationTree) -> List[str]:
    result: List[str] = []
    stack: List[isla.DerivationTree] = [tree]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)
            result.append(f"in({child.id}, {node.id})")

    result.append("tin(A, B) :- in(A, B)")
    result.append("tin(A, C) :- in(A, B), tin(B, C)")

    return result


def evaluate_prolog_query(assumptions: List[str], query: str) -> Optional[List[Dict[str, int]]]:
    with PrologMQI() as mqi:
        with mqi.create_thread() as prolog_thread:
            for assumption in assumptions:
                prolog_thread.query(f"assert(({assumption}))")
            result = prolog_thread.query(query)
            if result is False:
                return None

            return result


def split_cnf(formula: isla.Formula) -> Optional[List[isla.Formula]]:
    if (isinstance(formula, isla.StructuralPredicateFormula) or
            isinstance(formula, isla.SemanticPredicateFormula) or
            isinstance(formula, isla.SMTFormula)):
        return [formula]

    if isinstance(formula, isla.ConjunctiveFormula):
        result: List[isla.Formula] = []
        for child in formula.args:
            child_result = split_cnf(child)
            if child is None:
                return None
            result.extend(child_result)
        return result

    return None


def get_placeholders(formula: isla.Formula) -> Set[PlaceholderVariable]:
    placeholders = {var for var in isla.VariablesCollector.collect(formula)
                    if isinstance(var, PlaceholderVariable)}
    assert all(isinstance(ph, NonterminalPlaceholderVariable) or
               isinstance(ph, NonterminalStringPlaceholderVariable) for ph in placeholders), \
        "Only NonterminalPlaceholderVariables or NonterminalStringPlaceholderVariables supported so far."
    return placeholders


def get_quantifier_free_core(formula: isla.Formula) -> Optional[isla.Formula]:
    if isinstance(formula, isla.QuantifiedFormula):
        return get_quantifier_free_core(formula.inner_formula)
    if isinstance(formula, isla.IntroduceNumericConstantFormula):
        return get_quantifier_free_core(formula.inner_formula)

    visitor = isla.FilterVisitor(
        lambda f: (isinstance(f, isla.QuantifiedFormula) or
                   isinstance(f, isla.IntroduceNumericConstantFormula)))
    if visitor.collect(formula):
        return None

    return formula


def extract_top_level_constant(candidate):
    return next(
        (c for c in isla.VariablesCollector.collect(candidate)
         if isinstance(c, isla.Constant) and not c.is_numeric()))


class PrologTranslator(ABC):
    def responsible(self, formula: isla.Formula) -> bool:
        raise NotImplementedError()

    def query(self, formula: isla.Formula, var_map: Dict[PlaceholderVariable, str]) -> str:
        raise NotImplementedError()

    def facts(self, tree: isla.DerivationTree) -> List[str]:
        raise NotImplementedError()


class BeforePredicateTranslator(PrologTranslator):
    def responsible(self, formula: isla.Formula) -> bool:
        return isinstance(formula, isla.StructuralPredicateFormula) and formula.predicate == BEFORE_PREDICATE

    def query(
            self,
            formula: isla.StructuralPredicateFormula,
            var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        arg_1 = formula.args[0]
        assert isinstance(arg_1, NonterminalPlaceholderVariable)
        arg_2 = formula.args[1]
        assert isinstance(arg_2, NonterminalPlaceholderVariable)
        return f"before({var_map[arg_1]}, {var_map[arg_2]})"

    def facts(self, tree: isla.DerivationTree) -> List[str]:
        ordered_trees: List[Tuple[isla.DerivationTree, isla.DerivationTree]] = [
            (t1, t2) for (p1, t1), (p2, t2)
            in itertools.product(*[[(path, tree) for path, tree in tree.paths()] for _ in range(2)])
            if is_before(None, p1, p2)]

        result: List[str] = []
        for t1, t2 in ordered_trees:
            result.append(f"before({t1.id}, {t2.id})")

        return result

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, BeforePredicateTranslator)


class VariablesEqualTranslator(PrologTranslator):
    def responsible(self, formula: isla.Formula) -> bool:
        return (isinstance(formula, isla.SMTFormula) and
                formula.formula.decl().kind() == z3.Z3_OP_EQ and
                all(is_z3_var(child) for child in formula.formula.children()))

    def query(self, formula: isla.SMTFormula, var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        free_vars: List[NonterminalPlaceholderVariable] = cast(
            List[NonterminalPlaceholderVariable], list(formula.free_variables()))
        assert len(free_vars) == 2
        assert all(isinstance(var, NonterminalPlaceholderVariable) for var in free_vars)

        return f"eq({var_map[free_vars[0]]}, {var_map[free_vars[1]]})"

    def facts(self, tree: isla.DerivationTree) -> List[str]:
        equal_trees: List[Tuple[isla.DerivationTree, isla.DerivationTree]] = [
            (t1, t2) for (p1, t1), (p2, t2)
            in itertools.product(*[[(path, tree) for path, tree in tree.paths()] for _ in range(2)])
            if str(t1) == str(t2)]

        return [f"eq({t1.id}, {t2.id})" for t1, t2 in equal_trees]

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, VariablesEqualTranslator)


class CountPredicateTranslator(PrologTranslator):
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.graph = gg.GrammarGraph.from_grammar(grammar)

    def reachable(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return reachable(self.graph, from_nonterminal, to_nonterminal)

    def responsible(self, formula: isla.Formula) -> bool:
        return isinstance(formula, isla.SemanticPredicateFormula) and formula.predicate == COUNT_PREDICATE

    def query(
            self,
            formula: isla.SemanticPredicateFormula,
            var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        args = [
            var_map[arg] if isinstance(arg, PlaceholderVariable) else
            (f'"{arg}"' if isinstance(arg, str)
             else "_")
            for arg in formula.args]
        return f'count({", ".join(args)})'

    def facts(self, tree: isla.DerivationTree) -> List[str]:
        result: List[str] = []

        for _, subtree in tree.paths():
            if not subtree.children:
                continue
            assert is_nonterminal(subtree.value)

            # Only consider subtrees with nonterminal symbols that can occur a variable amount
            # of times in derivations form the grammar.
            if all(not self.reachable(other_nonterminal, other_nonterminal) or
                   not self.reachable(other_nonterminal, subtree.value)
                   for other_nonterminal in self.grammar):
                continue

            occurrences: Dict[str, int] = {}

            for nonterminal in self.grammar:
                # Consider only nonterminals that are reachable from this subtree.
                if not self.reachable(subtree.value, nonterminal):
                    continue

                # Furthermore, consider only those which can occur different numbers of times in derivations from
                # the subgrammar defined by the nonterminal of this subtree. This is the case if there is some
                # recursive other nonterminal reachable form the nonterminal of this subtree, from which this
                # nonterminal can be reached.
                if any(self.reachable(subtree.value, other_nonterminal) and
                       self.reachable(other_nonterminal, other_nonterminal) and
                       self.reachable(other_nonterminal, nonterminal)
                       for other_nonterminal in self.grammar):
                    occurrences[nonterminal] = 0

            def action(_: isla.Path, subsubtree: isla.DerivationTree):
                nonlocal occurrences
                if subsubtree.children and subsubtree.value in occurrences:
                    occurrences[subsubtree.value] += 1

            subtree.traverse(action)

            result.extend([
                f'count({subtree.id}, "{nonterminal}", {occs})'
                for nonterminal, occs in occurrences.items()])

        return result

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, CountPredicateTranslator)


def get_translator(formula: isla.Formula, grammar: Grammar) -> Optional[PrologTranslator]:
    translators = [
        BeforePredicateTranslator(),
        CountPredicateTranslator(grammar),
        VariablesEqualTranslator(),
    ]

    try:
        return next(translator for translator in translators
                    if translator.responsible(formula))
    except StopIteration:
        return None
