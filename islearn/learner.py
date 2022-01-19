import itertools
import logging
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Optional, cast

import z3
from grammar_graph import gg
from isla import isla
from isla.helpers import is_z3_var
from isla.isla_predicates import is_before, BEFORE_PREDICATE
from isla.solver import get_quantifier_chains
from isla.type_defs import Grammar
from swiplserver import PrologMQI

from islearn.pattern import Pattern, Placeholders

logger = logging.getLogger("learner")


def filter_invariants(
        patterns: List[isla.Formula],
        inputs: List[isla.DerivationTree],
        grammar: Grammar) -> List[isla.Formula]:
    candidates: List[isla.Formula] = []
    for pattern in patterns:
        constant: isla.Constant = extract_top_level_constant(pattern)

        for inp in inputs:
            assumptions: List[str] = []

            assumptions.extend([f"inner_node({tree.id})" for _, tree in inp.paths() if tree.num_children() > 0])

            stack: List[isla.DerivationTree] = [inp]
            while stack:
                node = stack.pop()
                for child in node.children:
                    stack.append(child)
                    assumptions.append(f"in({child.id}, {node.id})")

            ordered_trees: List[Tuple[isla.DerivationTree, isla.DerivationTree]] = [
                (t1, t2) for (p1, t1), (p2, t2)
                in itertools.product(*[[(path, tree) for path, tree in inp.paths()] for _ in range(2)])
                if is_before(None, p1, p2)]

            for t1, t2 in ordered_trees:
                assumptions.append(f"before({t1.id}, {t2.id})")

            equal_trees: List[Tuple[isla.DerivationTree, isla.DerivationTree]] = [
                (t1, t2) for (p1, t1), (p2, t2)
                in itertools.product(*[[(path, tree) for path, tree in inp.paths()] for _ in range(2)])
                if str(t1) == str(t2)]

            for t1, t2 in equal_trees:
                assumptions.append(f"eq({t1.id}, {t2.id})")

            assumptions.append("tin(A, B) :- in(A, B)")
            assumptions.append("tin(A, C) :- in(A, B), tin(B, C)")

            placeholders = sorted(list(get_placeholders(pattern)))
            var_map: Dict[NonterminalPlaceholderVariable, str] = {
                placeholder: placeholder.name.upper()
                for placeholder in placeholders}

            query = [f"inner_node({var})" for var in var_map.values()]

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
                        iv = inp.id
                    else:
                        assert isinstance(formula.in_variable, NonterminalPlaceholderVariable)
                        iv = var_map[formula.in_variable]

                    query.append(f"tin({var_map[bv]}, {iv})")

            pattern.accept(InVisitor())

            core = get_quantifier_free_core(pattern)
            assert core is not None, "Only supporting formulas with one quantifier chain."

            conjunctive_formulas = split_cnf(core)
            assert conjunctive_formulas is not None, "Only supporting conjunctive quantifier-free core."

            for conjunct in conjunctive_formulas:
                if isinstance(conjunct, isla.StructuralPredicateFormula) and conjunct.predicate == BEFORE_PREDICATE:
                    arg_1 = conjunct.args[0]
                    assert isinstance(arg_1, NonterminalPlaceholderVariable)
                    arg_2 = conjunct.args[1]
                    assert isinstance(arg_2, NonterminalPlaceholderVariable)
                    query.append(f"before({var_map[arg_1]}, {var_map[arg_2]})")
                elif isinstance(conjunct, isla.SMTFormula):
                    inner: z3.BoolRef = conjunct.formula
                    if inner.decl().kind() == z3.Z3_OP_EQ and all(is_z3_var(child) for child in inner.children()):
                        free_vars: List[NonterminalPlaceholderVariable] = cast(
                            List[NonterminalPlaceholderVariable], list(conjunct.free_variables()))
                        assert len(free_vars) == 2
                        assert all(isinstance(var, NonterminalPlaceholderVariable) for var in free_vars)
                        query.append(f"eq({var_map[free_vars[0]]}, {var_map[free_vars[1]]})")

            result = evaluate_prolog_query(assumptions, ", ".join(query))
            if result is None:
                continue

            for instantiation in result:
                candidates.append(pattern.substitute_variables({
                    placeholder: isla.BoundVariable(
                        placeholder.name,
                        inp.get_subtree(inp.find_node(instantiation[variable])).value)
                    for placeholder, variable in var_map.items()}))

    return [
        candidate for candidate in candidates
        if all(isla.evaluate(candidate, inp, grammar) for inp in inputs)
    ]


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


class PlaceholderVariable(isla.BoundVariable, ABC):
    pass


@dataclass(frozen=True, init=True)
class NonterminalPlaceholderVariable(PlaceholderVariable):
    name: str


def get_placeholders(formula: isla.Formula) -> Set[NonterminalPlaceholderVariable]:
    placeholders = {var for var in isla.VariablesCollector.collect(formula)
                    if isinstance(var, PlaceholderVariable)}
    assert all(isinstance(ph, NonterminalPlaceholderVariable) for ph in placeholders), \
        "Only NonterminalPlaceholderVariables supported so far."
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


def check_vacuous_satisfaction(
        formula: isla.Formula,
        vacuously_matched_quantifiers: Set[isla.ForallFormula]) -> bool:
    if not isla.get_toplevel_quantified_formulas(formula) or not vacuously_matched_quantifiers:
        return False

    # TODO: Deal with conjunctions / disjunctions and v.s. in only one part.
    quantifier_chains: List[Tuple[isla.ForallFormula, ...]] = [
        tuple([f for f in c if isinstance(f, isla.ForallFormula)])
        for c in get_quantifier_chains(formula)]
    quantifier_chains = [c for c in quantifier_chains if c]

    vacuous_chains = {
        c for c in quantifier_chains if
        any(any(of.id == f.id for of in vacuously_matched_quantifiers)
            for f in c)}

    assert len(vacuous_chains) <= len(quantifier_chains)
    if len(vacuous_chains) < len(quantifier_chains):
        return False

    return True
