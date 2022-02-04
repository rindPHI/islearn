import io
import itertools
import logging
import os.path
import pkgutil
from abc import ABC
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable

import isla.fuzzer
import z3
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.helpers import is_z3_var, is_nonterminal
from isla.isla_predicates import is_before, BEFORE_PREDICATE, COUNT_PREDICATE, reachable
from isla.type_defs import Grammar
from swiplserver import PrologMQI

from islearn.helpers import e_assert_present
from islearn.language import NonterminalPlaceholderVariable, PlaceholderVariable, NonterminalStringPlaceholderVariable, \
    parse_abstract_isla

STANDARD_PATTERNS_REPO = "patterns.yaml"
logger = logging.getLogger("learner")


def learn_invariants(
        grammar: Grammar,
        prop: Callable[[language.DerivationTree], bool],
        positive_examples: Optional[Iterable[language.DerivationTree]] = None,
        negative_examples: Optional[Iterable[language.DerivationTree]] = None,
        patterns: Optional[List[language.Formula | str]] = None,
        pattern_file: Optional[str] = None,
        activated_patterns: Optional[Iterable[str]] = None,
        deactivated_patterns: Optional[Iterable[str]] = None) -> List[language.Formula]:
    positive_examples = set(positive_examples or [])
    negative_examples = set(negative_examples or [])

    assert all(prop(example) for example in positive_examples)
    assert all(not prop(example) for example in negative_examples)

    if not patterns:
        pattern_repo = patterns_from_file(pattern_file or STANDARD_PATTERNS_REPO)
        if activated_patterns:
            patterns = [pattern for name, pattern in pattern_repo.items() if name in activated_patterns]
        elif deactivated_patterns:
            patterns = [pattern for name, pattern in pattern_repo.items() if name not in deactivated_patterns]
        else:
            patterns = list(pattern_repo.values())
    else:
        patterns = [
            pattern if isinstance(pattern, language.Formula)
            else parse_abstract_isla(pattern, grammar)
            for pattern in patterns]

    # TODO: Also consider inverted patterns.

    fuzzer = isla.fuzzer.GrammarCoverageFuzzer(grammar)
    desired_number_examples = 10
    num_tries = 100
    if len(positive_examples) < desired_number_examples or len(negative_examples) < desired_number_examples:
        i = 0
        while ((len(positive_examples) < desired_number_examples
                or len(negative_examples) < desired_number_examples)
               and i < num_tries):
            i += 1
            inp = fuzzer.expand_tree(language.DerivationTree("<start>", None))
            if prop(inp):
                positive_examples.add(inp)
            else:
                negative_examples.add(inp)

    invs_1 = filter_invariants(patterns, positive_examples, grammar)

    invs_2 = [
        inv for inv in invs_1
        if not any(evaluate(inv, inp, grammar).is_true()
                   for inp in negative_examples)]

    # TODO: Learn invariants for negative samples, invert them?

    return invs_2


def filter_invariants(
        patterns: Iterable[language.Formula | str],
        inputs: Iterable[language.DerivationTree],
        grammar: Grammar) -> List[language.Formula]:
    patterns = [
        pattern if isinstance(pattern, language.Formula)
        else parse_abstract_isla(pattern, grammar)
        for pattern in patterns]

    candidates: Set[language.Formula] = set()

    for inp in inputs:
        assumptions: Set[str] = set()

        assumptions.update({f"inner_node({tree.id})" for _, tree in inp.paths() if tree.num_children() > 0})
        assumptions.update(get_in_assumptions(inp))

        all_conjunctive_formulas: List[language.Formula] = [
            f for pattern in patterns
            for f in e_assert_present(
                split_cnf(
                    e_assert_present(
                        get_quantifier_free_core(pattern),
                        "Only supporting formulas with one quantifier chain.")),
                "Only supporting conjunctive quantifier-free core.")
        ]

        translators: Set[PrologTranslator] = set(
            filter(
                lambda e: e is not None,
                [get_translator(formula, grammar)
                 for formula in all_conjunctive_formulas]))

        for translator in translators:
            assumptions.update(translator.facts(inp))

        for pattern in patterns:
            extended_assumptions: Set[str] = set(assumptions)
            var_map: Dict[PlaceholderVariable, str] = {
                placeholder: placeholder.name.upper()
                for placeholder in get_placeholders(pattern)}

            conjunctive_formulas = e_assert_present(
                split_cnf(
                    e_assert_present(
                        get_quantifier_free_core(pattern),
                        "Only supporting formulas with one quantifier chain.")),
                "Only supporting conjunctive quantifier-free core.")

            query = [f"inner_node({var})" for ph, var in var_map.items()
                     if isinstance(ph, NonterminalPlaceholderVariable)]
            query.extend(get_in_query(pattern, inp.id, var_map))

            query.extend([
                translator.query(conjunct, var_map)
                for conjunct in conjunctive_formulas
                for translator in translators
                if translator.responsible(conjunct)])

            if any(isinstance(var, NonterminalStringPlaceholderVariable) for var in var_map):
                extended_assumptions.update({f"nonterminal(\"{nonterminal}\")" for nonterminal in grammar})
                query.extend([f"nonterminal({var_map[var]})"
                              for var in var_map
                              if isinstance(var, NonterminalStringPlaceholderVariable)])

            result = evaluate_prolog_query(extended_assumptions, ", ".join(query))
            if result is None:
                continue

            for instantiation in result:
                candidates.add(pattern.substitute_variables({
                    placeholder:
                        language.BoundVariable(
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
        if all(evaluate(candidate, inp, grammar) for inp in inputs)
    ]


def get_in_query(
        formula: language.Formula,
        root_id: int,
        var_map: Dict[PlaceholderVariable, str]) -> List[str]:
    query: List[str] = []
    constant = extract_top_level_constant(formula)

    class InVisitor(language.FormulaVisitor):
        def visit_exists_formula(self, formula: language.ExistsFormula):
            self.handle(formula)

        def visit_forall_formula(self, formula: language.ForallFormula):
            self.handle(formula)

        def handle(self, formula: language.QuantifiedFormula):
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


def get_in_assumptions(tree: language.DerivationTree) -> List[str]:
    result: List[str] = []
    stack: List[language.DerivationTree] = [tree]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)
            result.append(f"in({child.id}, {node.id})")

    result.append("tin(A, B) :- in(A, B)")
    result.append("tin(A, C) :- in(A, B), tin(B, C)")

    return result


def evaluate_prolog_query(assumptions: Set[str], query: str) -> Optional[List[Dict[str, int]]]:
    with PrologMQI() as mqi:
        with mqi.create_thread() as prolog_thread:
            for assumption in assumptions:
                prolog_thread.query(f"assert(({assumption}))")
            result = prolog_thread.query(query)
            if result is False:
                return None

            return result


def split_cnf(formula: language.Formula) -> Optional[List[language.Formula]]:
    if (isinstance(formula, language.StructuralPredicateFormula) or
            isinstance(formula, language.SemanticPredicateFormula) or
            isinstance(formula, language.SMTFormula)):
        return [formula]

    if isinstance(formula, language.ConjunctiveFormula):
        result: List[language.Formula] = []
        for child in formula.args:
            child_result = split_cnf(child)
            if child is None:
                return None
            result.extend(child_result)
        return result

    return None


def get_placeholders(formula: language.Formula) -> Set[PlaceholderVariable]:
    placeholders = {var for var in language.VariablesCollector.collect(formula)
                    if isinstance(var, PlaceholderVariable)}
    assert all(isinstance(ph, NonterminalPlaceholderVariable) or
               isinstance(ph, NonterminalStringPlaceholderVariable) for ph in placeholders), \
        "Only NonterminalPlaceholderVariables or NonterminalStringPlaceholderVariables supported so far."
    return placeholders


def get_quantifier_free_core(formula: language.Formula) -> Optional[language.Formula]:
    if isinstance(formula, language.QuantifiedFormula):
        return get_quantifier_free_core(formula.inner_formula)
    if isinstance(formula, language.NumericQuantifiedFormula):
        return get_quantifier_free_core(formula.inner_formula)

    visitor = language.FilterVisitor(
        lambda f: (isinstance(f, language.QuantifiedFormula) or
                   isinstance(f, language.NumericQuantifiedFormula)))
    if visitor.collect(formula):
        return None

    return formula


def extract_top_level_constant(candidate):
    return next(
        (c for c in language.VariablesCollector.collect(candidate)
         if isinstance(c, language.Constant) and not c.is_numeric()))


class PrologTranslator(ABC):
    def responsible(self, formula: language.Formula) -> bool:
        raise NotImplementedError()

    def query(self, formula: language.Formula, var_map: Dict[PlaceholderVariable, str]) -> str:
        raise NotImplementedError()

    def facts(self, tree: language.DerivationTree) -> List[str]:
        raise NotImplementedError()


class BeforePredicateTranslator(PrologTranslator):
    def responsible(self, formula: language.Formula) -> bool:
        return isinstance(formula, language.StructuralPredicateFormula) and formula.predicate == BEFORE_PREDICATE

    def query(
            self,
            formula: language.StructuralPredicateFormula,
            var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        arg_1 = formula.args[0]
        assert isinstance(arg_1, NonterminalPlaceholderVariable)
        arg_2 = formula.args[1]
        assert isinstance(arg_2, NonterminalPlaceholderVariable)
        return f"before({var_map[arg_1]}, {var_map[arg_2]})"

    def facts(self, tree: language.DerivationTree) -> List[str]:
        ordered_trees: List[Tuple[language.DerivationTree, language.DerivationTree]] = [
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
    def responsible(self, formula: language.Formula) -> bool:
        return (isinstance(formula, language.SMTFormula) and
                formula.formula.decl().kind() == z3.Z3_OP_EQ and
                all(is_z3_var(child) for child in formula.formula.children()))

    def query(self, formula: language.SMTFormula, var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        free_vars: List[NonterminalPlaceholderVariable] = cast(
            List[NonterminalPlaceholderVariable], list(formula.free_variables()))
        assert len(free_vars) == 2
        assert all(isinstance(var, NonterminalPlaceholderVariable) for var in free_vars)

        return f"eq({var_map[free_vars[0]]}, {var_map[free_vars[1]]})"

    def facts(self, tree: language.DerivationTree) -> List[str]:
        equal_trees: List[Tuple[language.DerivationTree, language.DerivationTree]] = [
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

    def responsible(self, formula: language.Formula) -> bool:
        return isinstance(formula, language.SemanticPredicateFormula) and formula.predicate == COUNT_PREDICATE

    def query(
            self,
            formula: language.SemanticPredicateFormula,
            var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        args = [
            var_map[arg] if isinstance(arg, PlaceholderVariable) else
            (f'"{arg}"' if isinstance(arg, str)
             else "_")
            for arg in formula.args]
        return f'count({", ".join(args)})'

    def facts(self, tree: language.DerivationTree) -> List[str]:
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

            def action(_: language.Path, subsubtree: language.DerivationTree):
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


def get_translator(formula: language.Formula, grammar: Grammar) -> Optional[PrologTranslator]:
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


def patterns_from_file(file_name: str = STANDARD_PATTERNS_REPO) -> Dict[str, language.Formula]:
    from yaml import load
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    if os.path.isfile(file_name):
        f = open(file_name, "r")
        contents = f.read()
        f.close()
    else:
        contents = pkgutil.get_data("islearn", STANDARD_PATTERNS_REPO).decode("UTF-8")

    data = load(io.StringIO(contents), Loader=Loader)
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(entry, dict) for entry in data)

    return {
        entry["name"]: parse_abstract_isla(entry["constraint"])
        for entry in data}
