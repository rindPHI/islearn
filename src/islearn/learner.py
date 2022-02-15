import functools
import io

from isla.language import set_smt_auto_eval
from isla.solver import ISLaSolver
from pathos import multiprocessing as pmp
import itertools
import logging
import os.path
import pkgutil
from abc import ABC, ABCMeta
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable

import isla.fuzzer
import z3
from grammar_graph import gg
from isla import language, isla_predicates
from isla.evaluator import evaluate
from isla.helpers import is_z3_var, is_nonterminal, z3_subst, dict_of_lists_to_list_of_dicts
from isla.isla_predicates import is_before, BEFORE_PREDICATE, COUNT_PREDICATE, reachable
from isla.type_defs import Grammar
from swiplserver import PrologMQI

from islearn.helpers import e_assert_present, parallel_all, parallel_any, transitive_closure, mappings, \
    connected_chains, non_consecutive_ordered_sub_sequences, replace_formula_by_formulas
from islearn.language import NonterminalPlaceholderVariable, PlaceholderVariable, NonterminalStringPlaceholderVariable, \
    parse_abstract_isla, StringPlaceholderVariable, AbstractISLaUnparser
from islearn.mutation import MutationFuzzer

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
        deactivated_patterns: Optional[Iterable[str]] = None,
        k: int = 3) -> Dict[language.Formula, float]:
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

    logger.info(
        "Starting with %d positive, and %d negative samples.",
        len(positive_examples),
        len(negative_examples)
    )

    ne_before = len(negative_examples)
    pe_before = len(positive_examples)
    generate_sample_inputs(grammar, prop, negative_examples, positive_examples)

    logger.info(
        "Generated %d additional positive, and %d additional negative samples (from scratch).",
        len(positive_examples) - pe_before,
        len(negative_examples) - ne_before
    )

    assert len(positive_examples) > 0, "Cannot learn without any positive examples!"

    pe_before = len(positive_examples)
    ne_before = len(negative_examples)
    mutation_fuzzer = MutationFuzzer(grammar, positive_examples, prop, k=k)
    for inp in mutation_fuzzer.run(num_iterations=50, alpha=.1, yield_negative=True):
        if prop(inp):
            positive_examples.add(inp)
        else:
            negative_examples.add(inp)

    logger.info(
        "Generated %d additional positive, and %d additional negative samples (by mutation fuzzing).",
        len(positive_examples) - pe_before,
        len(negative_examples) - ne_before
    )

    graph = gg.GrammarGraph.from_grammar(grammar)
    positive_examples = filter_inputs_by_paths(positive_examples, graph, max_cnt=10, k=k)
    positive_examples_for_learning = filter_inputs_by_paths(positive_examples, graph, max_cnt=2, k=k, prefer_small=True)
    negative_examples = filter_inputs_by_paths(negative_examples, graph, max_cnt=10, k=k)

    logger.info(
        "Reduced positive / negative samples to subsets of %d / %d samples based on k-path coverage, "
        "keeping %d positive examples for candidate generation.",
        len(positive_examples),
        len(negative_examples),
        len(positive_examples_for_learning),
    )

    candidates = generate_candidates(patterns, positive_examples_for_learning, grammar)
    logger.info("Found %d invariant candidates", len(candidates))

    # Only consider *real* invariants
    invariants = [
        candidate for candidate in candidates
        if parallel_all(lambda inp: evaluate(candidate, inp, grammar).is_true(), positive_examples)
    ]

    logger.info("%d invariants remain after filtering", len(invariants))

    # ne_before = len(negative_examples)
    # negative_examples.update(generate_counter_examples_from_formulas(grammar, prop, invariants))
    # logger.info(
    #     "Generated %d additional negative samples (from invariants).",
    #     len(negative_examples) - ne_before
    # )

    logger.info("Calculating precision")

    with pmp.ProcessingPool(processes=2 * pmp.cpu_count()) as pool:
        eval_results = pool.map(
            lambda t: (t[0], int(evaluate(t[0], t[1], grammar).is_true())),
            itertools.product(invariants, negative_examples),
            chunksize=10
        )

    result: Dict[language.Formula, float] = {
        inv: 1 - (sum([eval_result for other_inv, eval_result in eval_results if other_inv == inv])
                  / len(negative_examples))
        for inv in invariants
    }

    # result: Dict[language.Formula, float] = {}
    # for invariant in invariants:
    #   with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
    #       num_counter_examples = sum(pool.map(
    #           lambda negative_example: int(evaluate(invariant, negative_example, grammar).is_true()),
    #           negative_examples))

    #   result[invariant] = 1 - (num_counter_examples / (len(negative_examples) or 1))

    logger.info("Done.")

    return dict(cast(List[Tuple[language.Formula, float]],
                     sorted(result.items(), key=lambda p: p[1], reverse=True)))


def generate_sample_inputs(
        grammar: Grammar,
        prop: Callable[[language.DerivationTree], bool],
        negative_examples: Set[language.DerivationTree],
        positive_examples: Set[language.DerivationTree],
        desired_number_examples: int = 10,
        num_tries: int = 100) -> None:
    fuzzer = isla.fuzzer.GrammarCoverageFuzzer(grammar)
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


def generate_counter_examples_from_formulas(
        grammar: Grammar,
        prop: Callable[[language.DerivationTree], bool],
        formulas: Iterable[language.Formula],
        desired_number_counter_examples: int = 50,
        num_tries: int = 100) -> Set[language.DerivationTree]:
    result: Set[language.DerivationTree] = set()

    solvers = {
        formula: ISLaSolver(grammar, formula, enforce_unique_trees_in_queue=False).solve()
        for formula in formulas}

    i = 0
    while len(result) < desired_number_counter_examples and i < num_tries:
        # with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
        #     result.update({
        #         inp
        #         for inp in e_assert_present(pool.map(lambda f: next(solvers[f]), formulas))
        #         if not prop(inp)
        #     })
        for formula in formulas:
            for _ in range(desired_number_counter_examples):
                inp = next(solvers[formula])
                if not prop(inp):
                    result.add(inp)

        i += 1

    return result


def filter_inputs_by_paths(
        inputs: Iterable[language.DerivationTree],
        graph: gg.GrammarGraph,
        max_cnt: int = 10,
        k: int = 3,
        prefer_small=False) -> Set[language.DerivationTree]:
    inputs = set(inputs)

    if len(inputs) <= max_cnt:
        return inputs

    tree_paths = {inp: graph.k_paths_in_tree(inp.to_parse_tree(), k) for inp in inputs}

    result: Set[language.DerivationTree] = set([])
    covered_paths: Set[Tuple[gg.Node, ...]] = set([])

    def uncovered_paths(inp: language.DerivationTree) -> Set[Tuple[gg.Node, ...]]:
        return {path for path in tree_paths[inp] if path not in covered_paths}

    while inputs and len(result) < max_cnt and len(covered_paths) < len(graph.k_paths(k)):
        if prefer_small:
            try:
                inp = next(inp for inp in sorted(inputs, key=lambda inp: len(inp))
                           if uncovered_paths(inp))
            except StopIteration:
                break
        else:
            inp = sorted(inputs, key=lambda inp: (len(uncovered_paths(inp)), -len(inp)), reverse=True)[0]

        covered_paths.update(tree_paths[inp])
        inputs.remove(inp)
        result.add(inp)

    return result


def generate_candidates(
        patterns: Iterable[language.Formula | str],
        inputs: Iterable[language.DerivationTree],
        grammar: Grammar) -> Set[language.Formula]:
    graph = gg.GrammarGraph.from_grammar(grammar)

    logger.debug("Computing nonterminal chains in inputs.")
    nonterminal_paths: Set[Tuple[str, ...]] = {
        tuple([inp.get_subtree(path[:idx]).value
               for idx in range(len(path))])
        for inp in inputs
        for path, _ in inp.leaves()
    }

    nonterminal_chains: Set[Tuple[str, ...]] = {
        tuple(reversed(path)) for path in nonterminal_paths
        if not any(len(path_) > len(path) and path_[:len(path)] == path for path_ in nonterminal_paths)
    }
    assert all(chain[-1] == "<start>" for chain in nonterminal_chains)
    logger.debug("Found %d nonterminal chains.", len(nonterminal_chains))

    filters: List[PatternInstantiationFilter] = [
        StructuralPredicatesFilter(),
        VariablesEqualFilter(),
        NonterminalStringInCountPredicatesFilter(graph, nonterminal_chains)
    ]

    patterns = [
        parse_abstract_isla(pattern, grammar) if isinstance(pattern, str)
        else pattern
        for pattern in patterns]

    result: Set[language.Formula] = set([])

    for pattern in patterns:
        logger.debug("Instantiating pattern\n%s", AbstractISLaUnparser(pattern).unparse())
        set_smt_auto_eval(pattern, False)

        # Instantiate various placeholder variables:
        # 1. Nonterminal placeholders
        pattern_insts_without_nonterminal_placeholders = instantiate_nonterminal_placeholders(
            pattern, nonterminal_chains)

        logger.debug("Found %d instantiations of pattern meeting quantifier requirements",
                     len(pattern_insts_without_nonterminal_placeholders))

        # 2. Nonterminal-String placeholders
        pattern_insts_without_nonterminal_string_placeholders = instantiate_nonterminal_string_placeholders(
            grammar, pattern_insts_without_nonterminal_placeholders)

        logger.debug("Found %d instantiations of pattern after instantiating nonterminal string placeholders",
                     len(pattern_insts_without_nonterminal_string_placeholders))

        # 3. String placeholders
        pattern_insts_without_string_placeholders = instantiate_string_placeholders(
            pattern_insts_without_nonterminal_string_placeholders, inputs)

        logger.debug("Found %d instantiations of pattern after instantiating string placeholders",
                     len(pattern_insts_without_string_placeholders))

        assert all(not get_placeholders(candidate)
                   for candidate in pattern_insts_without_string_placeholders)

        pattern_insts_meeting_atom_requirements: Set[language.Formula] = \
            set(pattern_insts_without_string_placeholders)

        for pattern_filter in filters:
            pattern_insts_meeting_atom_requirements = {
                pattern_inst for pattern_inst in pattern_insts_meeting_atom_requirements
                if pattern_filter.predicate(pattern_inst, inputs)
            }

            logger.debug("%d instantiations remaining after filter '%s'",
                         len(pattern_insts_meeting_atom_requirements),
                         pattern_filter.name)

        result.update(pattern_insts_meeting_atom_requirements)

    return result


def instantiate_nonterminal_placeholders(
        pattern: language.Formula,
        nonterminal_chains: Set[Tuple[str, ...]]) -> Set[language.Formula]:
    in_visitor = InVisitor()
    pattern.accept(in_visitor)
    variable_chains: Set[Tuple[language.Variable, ...]] = connected_chains(in_visitor.result)
    assert all(chain[-1] == extract_top_level_constant(pattern) for chain in variable_chains)

    instantiations: List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]] = []
    for variable_chain in variable_chains:
        nonterminal_sequences: Set[Tuple[str, ...]] = {
            sequence
            for nonterminal_chain in nonterminal_chains
            for sequence in non_consecutive_ordered_sub_sequences(nonterminal_chain[:-1], len(variable_chain) - 1)
        }

        new_instantiations = [
            {variable: language.BoundVariable(variable.name, nonterminal_sequence[idx])
             for idx, variable in enumerate(variable_chain[:-1])}
            for nonterminal_sequence in nonterminal_sequences]

        if not instantiations:
            instantiations = new_instantiations
        else:
            instantiations = [
                cast(Dict[NonterminalPlaceholderVariable, language.BoundVariable],
                     functools.reduce(dict.__or__, t))
                for t in list(itertools.product(instantiations, new_instantiations))
            ]
    pattern_insts_without_nonterminal_placeholders: Set[language.Formula] = {
        pattern.substitute_variables(instantiation)
        for instantiation in instantiations
    }
    return pattern_insts_without_nonterminal_placeholders


def instantiate_nonterminal_string_placeholders(
        grammar: Grammar,
        inst_patterns: Set[language.Formula]) -> Set[language.Formula]:
    if all(not isinstance(placeholder, NonterminalStringPlaceholderVariable)
           for inst_pattern in inst_patterns
           for placeholder in get_placeholders(inst_pattern)):
        return inst_patterns

    result: Set[language.Formula] = set([])
    for inst_pattern in inst_patterns:
        for nonterminal in grammar:
            def replace_placeholder_by_nonterminal_string(
                    subformula: language.Formula) -> language.Formula | bool:
                if (not isinstance(subformula, language.SemanticPredicateFormula) and
                        not isinstance(subformula, language.StructuralPredicateFormula)):
                    return False

                if not any(isinstance(arg, NonterminalStringPlaceholderVariable) for arg in subformula.args):
                    return False

                new_args = [
                    nonterminal if isinstance(arg, NonterminalStringPlaceholderVariable)
                    else arg
                    for arg in subformula.args]

                constructor = (
                    language.StructuralPredicateFormula
                    if isinstance(inst_pattern, language.StructuralPredicateFormula)
                    else language.SemanticPredicateFormula)

                return constructor(subformula.predicate, *new_args)

            result.add(language.replace_formula(inst_pattern, replace_placeholder_by_nonterminal_string))

    return result


def instantiate_string_placeholders(
        inst_patterns: Set[language.Formula],
        inputs: Iterable[language.DerivationTree]
) -> Set[language.Formula]:
    if all(not isinstance(placeholder, StringPlaceholderVariable)
           for inst_pattern in inst_patterns
           for placeholder in get_placeholders(inst_pattern)):
        return inst_patterns

    result: Set[language.Formula] = set([])
    for inst_pattern in inst_patterns:
        def replace_placeholder_by_string(subformula: language.Formula) -> Optional[Iterable[language.Formula]]:
            if (not isinstance(subformula, language.SemanticPredicateFormula) and
                    not isinstance(subformula, language.SemanticPredicateFormula) and
                    not isinstance(subformula, language.SMTFormula)):
                return None

            if not any(isinstance(arg, StringPlaceholderVariable) for arg in subformula.free_variables()):
                return None

            non_ph_vars = {v for v in subformula.free_variables() if not isinstance(v, PlaceholderVariable)}

            # We take all string representations of subtrees with the value of the nonterminal
            # types of the contained variables.
            insts = {
                str(tree)
                for inp in inputs
                for _, tree in inp.filter(lambda t: t.value in {v.n_type for v in non_ph_vars})}

            ph_vars = {v for v in subformula.free_variables() if isinstance(v, StringPlaceholderVariable)}
            sub_result: Set[language.Formula] = {subformula}
            for ph in ph_vars:
                old_sub_result = set(sub_result)
                sub_result = set([])
                for f in old_sub_result:
                    for inst in insts:
                        def substitute_string_placeholder(formula: language.Formula) -> language.Formula | bool:
                            if not any(v == ph for v in formula.free_variables()
                                       if isinstance(v, StringPlaceholderVariable)):
                                return False

                            if isinstance(formula, language.SMTFormula):
                                return language.SMTFormula(cast(z3.BoolRef, z3_subst(formula.formula, {
                                    ph.to_smt(): z3.StringVal(inst)
                                })), *[v for v in formula.free_variables() if v != ph])
                            elif isinstance(formula, language.StructuralPredicateFormula):
                                return language.StructuralPredicateFormula(
                                    formula.predicate,
                                    *[inst if arg == ph else arg for arg in formula.args]
                                )
                            elif isinstance(formula, language.SemanticPredicateFormula):
                                return language.SemanticPredicateFormula(
                                    formula.predicate,
                                    *[inst if arg == ph else arg for arg in formula.args]
                                )

                            return False

                        sub_result.add(language.replace_formula(f, substitute_string_placeholder))

            return sub_result

        result.update(replace_formula_by_formulas(inst_pattern, replace_placeholder_by_string))

    return result


class PatternInstantiationFilter(ABC):
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, PatternInstantiationFilter) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def predicate(self, formula: language.Formula, inputs: Iterable[language.DerivationTree]) -> bool:
        raise NotImplementedError()


class VariablesEqualFilter(PatternInstantiationFilter):
    def __init__(self):
        super().__init__("Variable Equality Filter")

    def predicate(self, formula: language.Formula, inputs: Iterable[language.DerivationTree]) -> bool:
        # We approximate satisfaction of constraints "var1 == var2" by checking whether there is
        # at least one input with two equal subtrees of nonterminal types matching those of the
        # variables.
        smt_equality_formulas: List[language.SMTFormula] = cast(
            List[language.SMTFormula],
            language.FilterVisitor(
                lambda f: (isinstance(f, language.SMTFormula) and
                           z3.is_eq(f.formula) and
                           len(f.free_variables()) == 2 and
                           all(is_z3_var(child) for child in f.formula.children()))
            ).collect(formula))

        if not smt_equality_formulas:
            return True

        for inp in inputs:
            success = True

            for smt_equality_formula in smt_equality_formulas:
                free_vars: List[NonterminalPlaceholderVariable] = cast(
                    List[NonterminalPlaceholderVariable],
                    list(smt_equality_formula.free_variables()))

                trees_1 = inp.filter(lambda t: t.value == free_vars[0].n_type)
                trees_2 = inp.filter(lambda t: t.value == free_vars[1].n_type)

                if not any(
                        t1.value == free_vars[0].n_type and
                        t2.value == free_vars[1].n_type
                        for (p1, t1), (p2, t2)
                        in itertools.product(trees_1, trees_2)
                        if str(t1) == str(t2) and p1 != p2):
                    success = False
                    break

            if success:
                return True

        return False


class StructuralPredicatesFilter(PatternInstantiationFilter):
    def __init__(self):
        super().__init__("Structural Predicates Filter")

    def predicate(self, formula: language.Formula, inputs: Iterable[language.DerivationTree]) -> bool:
        # We approximate satisfaction of structural predicate formulas by searching
        # inputs for subtrees with the right nonterminal types according to the argument
        # types of the structural formulas.

        structural_formulas: List[language.StructuralPredicateFormula] = cast(
            List[language.StructuralPredicateFormula],
            language.FilterVisitor(
                lambda f: isinstance(f, language.StructuralPredicateFormula)).collect(formula))

        if not structural_formulas:
            return True

        for inp in inputs:
            success = True
            for structural_formula in structural_formulas:
                arg_insts: Dict[language.Variable, Set[language.DerivationTree]] = {}
                for arg in structural_formula.free_variables():
                    arg_insts[arg] = {t for _, t in inp.filter(lambda subtree: subtree.value == arg.n_type)}
                arg_substitutions: List[Dict[language.Variable, language.DerivationTree]] = \
                    dict_of_lists_to_list_of_dicts(arg_insts)

                if not any(structural_formula.substitute_expressions(arg_substitution).evaluate(inp)
                           for arg_substitution in arg_substitutions):
                    success = False
                    break

            if success:
                return True

        return False


class NonterminalStringInCountPredicatesFilter(PatternInstantiationFilter):
    def __init__(self, graph: gg.GrammarGraph, nonterminal_chains: Set[Tuple[str, ...]]):
        super().__init__("Nonterminal String in `count` Predicates Filter")
        self.graph = graph
        self.nonterminal_chains = nonterminal_chains

    def reachable(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return reachable(self.graph, from_nonterminal, to_nonterminal)

    def reachable_in_inputs(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        from_nonterminal, to_nonterminal = to_nonterminal, from_nonterminal
        return any(
            from_nonterminal == nonterminal_1 and
            to_nonterminal == nonterminal_2
            for path in self.nonterminal_chains
            for idx_1, nonterminal_1 in enumerate(path)
            for idx_2, nonterminal_2 in enumerate(path)
            if idx_1 < idx_2
        )

    def predicate(self, formula: language.Formula, inputs: Iterable[language.DerivationTree]) -> bool:
        # In `count(elem, nonterminal, num)` occurrences
        # 1. the nonterminal must be reachable from the nonterminal type of elem. We
        #    consider reachability as defined by the sample inputs, not the grammar,
        #    to increase precision,
        # 2. it must be possible that the nonterminal occurs a variable number of times
        #    in the subgrammar of elem's nonterminal type.

        count_predicates: List[language.SemanticPredicateFormula] = cast(
            List[language.SemanticPredicateFormula],
            language.FilterVisitor(
                lambda f: isinstance(f, language.SemanticPredicateFormula) and
                          f.predicate == isla_predicates.COUNT_PREDICATE).collect(formula))

        if not count_predicates:
            return True

        def reachable_variable_number_of_times(start_nonterminal: str, needle_nonterminal: str) -> bool:
            return any(self.reachable_in_inputs(start_nonterminal, other_nonterminal) and
                       self.reachable(other_nonterminal, other_nonterminal) and
                       self.reachable_in_inputs(other_nonterminal, needle_nonterminal)
                       for other_nonterminal in self.graph.grammar)

        return any(
            reachable_variable_number_of_times(count_predicate.args[0].n_type, count_predicate.args[1])
            for count_predicate in count_predicates
        )


def generate_candidates_prolog(
        patterns: Iterable[str | language.Formula],
        inputs: Iterable[language.DerivationTree],
        grammar: Grammar) -> Set[language.Formula]:
    patterns = [
        parse_abstract_isla(pattern, grammar) if isinstance(pattern, str)
        else pattern
        for pattern in patterns]

    var_map: Dict[PlaceholderVariable, str] = {
        placeholder: placeholder.name.upper()
        for pattern in patterns
        for placeholder in get_placeholders(pattern)}

    conjunctive_formulas: Dict[language.Formula, List[language.Formula]] = {
        pattern: [
            f for f in e_assert_present(
                split_cnf(
                    e_assert_present(
                        get_quantifier_free_core(pattern),
                        "Only supporting formulas with one quantifier chain.")),
                "Only supporting conjunctive quantifier-free core.")]
        for pattern in patterns
    }

    translators: Set[PrologTranslator] = set(
        filter(
            lambda e: e is not None,
            [get_translator(formula, grammar)
             for conjuncts in conjunctive_formulas.values()
             for formula in conjuncts]))

    candidates: Set[language.Formula] = set()
    inp: language.DerivationTree
    pattern: language.Formula
    for inp, pattern in itertools.product(inputs, patterns):
        query = [f"inner_node({var})" for ph, var in var_map.items()
                 if isinstance(ph, NonterminalPlaceholderVariable)]
        query.extend(get_in_query(pattern, inp.id, var_map))
        query.extend([
            translator.query(conjunct, var_map)
            for conjunct in conjunctive_formulas[pattern]
            for translator in translators
            if translator.responsible(conjunct)])
        query.extend([f"nonterminal({var_map[var]})"
                      for var in var_map
                      if isinstance(var, NonterminalStringPlaceholderVariable)])

        result = evaluate_prolog_query(
            generate_assumptions(grammar, inp, translators),
            ", ".join(query))

        for instantiation in (result or []):
            candidate = pattern.substitute_variables({
                placeholder: language.BoundVariable(
                    placeholder.name,
                    inp.get_subtree(inp.find_node(instantiation[variable])).value)
                for placeholder, variable in var_map.items()
                if isinstance(placeholder, NonterminalPlaceholderVariable)})

            candidate = candidate.substitute_expressions({
                placeholder: instantiation[variable]
                for placeholder, variable in var_map.items()
                if isinstance(placeholder, NonterminalStringPlaceholderVariable)
            })

            def substitute_string_placeholders(formula: language.Formula) -> language.Formula | bool:
                if not isinstance(formula, language.SMTFormula):
                    return False

                string_placeholders = [
                    v for v in formula.free_variables()
                    if isinstance(v, StringPlaceholderVariable)]
                if not string_placeholders:
                    return False

                for ph in string_placeholders:
                    string_inst = instantiation[var_map[ph]]
                    assert isinstance(string_inst, str)

                    return language.SMTFormula(cast(z3.BoolRef, z3_subst(formula.formula, {
                        ph.to_smt(): z3.StringVal(string_inst)
                    })), *[v for v in formula.free_variables() if v not in string_placeholders])

            candidate = language.replace_formula(candidate, substitute_string_placeholders)

            candidates.add(candidate)

    return candidates


class PrologTranslator(ABC):
    def responsible(self, formula: language.Formula) -> bool:
        raise NotImplementedError()

    def query(self, formula: language.Formula, var_map: Dict[PlaceholderVariable, str]) -> str:
        raise NotImplementedError()

    def facts(self, tree: language.DerivationTree) -> Set[str]:
        raise NotImplementedError()


def generate_assumptions(
        grammar: Grammar,
        inp: language.DerivationTree,
        translators: Iterable[PrologTranslator]) -> Set[str]:
    assumptions: Set[str] = (
            {f"inner_node({tree.id})" for _, tree in inp.paths() if tree.num_children() > 0}
            | get_in_assumptions(inp)
            | {f"nonterminal(\"{nonterminal}\")" for nonterminal in grammar})

    for translator in translators:
        assumptions |= translator.facts(inp)

    return assumptions


class InVisitor(language.FormulaVisitor):
    def __init__(self):
        self.result: Set[Tuple[language.Variable, language.Variable]] = set()

    def visit_exists_formula(self, formula: language.ExistsFormula):
        self.handle(formula)

    def visit_forall_formula(self, formula: language.ForallFormula):
        self.handle(formula)

    def handle(self, formula: language.QuantifiedFormula):
        self.result.add((formula.bound_variable, formula.in_variable))


def get_in_query(
        formula: language.Formula,
        root_id: int,
        var_map: Dict[PlaceholderVariable, str]) -> List[str]:
    constant = extract_top_level_constant(formula)

    visitor = InVisitor()
    formula.accept(visitor)

    query: List[str] = []
    for bound_variable, in_variable in visitor.result:
        bv = cast(NonterminalPlaceholderVariable, bound_variable)
        assert isinstance(bv, NonterminalPlaceholderVariable)
        if in_variable == constant:
            iv = root_id
        else:
            assert isinstance(in_variable, NonterminalPlaceholderVariable)
            iv = var_map[in_variable]

        query.append(f"tin({var_map[bv]}, {iv})")

    return query


def get_in_assumptions(tree: language.DerivationTree) -> Set[str]:
    result: List[str] = []
    stack: List[language.DerivationTree] = [tree]
    while stack:
        node = stack.pop()
        for child in node.children:
            stack.append(child)
            result.append(f"in({child.id}, {node.id})")

    result.append("tin(A, B) :- in(A, B)")
    result.append("tin(A, C) :- in(A, B), tin(B, C)")

    return set(result)


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
    supported_placeholder_types = {
        NonterminalPlaceholderVariable,
        NonterminalStringPlaceholderVariable,
        StringPlaceholderVariable}

    assert all(any(isinstance(ph, t) for t in supported_placeholder_types) for ph in placeholders), \
        "Only " + ", ".join(map(lambda t: t.__name__, supported_placeholder_types)) + " supported so far."

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

    def facts(self, tree: language.DerivationTree) -> Set[str]:
        ordered_trees: List[Tuple[language.DerivationTree, language.DerivationTree]] = [
            (t1, t2) for (p1, t1), (p2, t2)
            in itertools.product(*[[(path, tree) for path, tree in tree.paths()] for _ in range(2)])
            if is_before(None, p1, p2)]

        result: List[str] = []
        for t1, t2 in ordered_trees:
            result.append(f"before({t1.id}, {t2.id})")

        return set(result)

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, BeforePredicateTranslator)


class VariablesEqualTranslator(PrologTranslator):
    def responsible(self, formula: language.Formula) -> bool:
        return (isinstance(formula, language.SMTFormula) and
                formula.formula.decl().kind() == z3.Z3_OP_EQ and
                len(formula.free_variables()) == 2 and
                all(isinstance(var, NonterminalPlaceholderVariable) for var in formula.free_variables()) and
                all(is_z3_var(child) for child in formula.formula.children()))

    def query(self, formula: language.SMTFormula, var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        free_vars: List[NonterminalPlaceholderVariable] = cast(
            List[NonterminalPlaceholderVariable], list(formula.free_variables()))

        return f"eqv({var_map[free_vars[0]]}, {var_map[free_vars[1]]})"

    def facts(self, tree: language.DerivationTree) -> Set[str]:
        equal_trees: List[Tuple[language.DerivationTree, language.DerivationTree]] = [
            (t1, t2) for (p1, t1), (p2, t2)
            in itertools.product(*[[(path, tree) for path, tree in tree.paths()] for _ in range(2)])
            if str(t1) == str(t2)]

        return {f"eqv({t1.id}, {t2.id})" for t1, t2 in equal_trees}

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, VariablesEqualTranslator)


class VariableEqualsStringTranslator(PrologTranslator):
    def responsible(self, formula: language.Formula) -> bool:
        return (isinstance(formula, language.SMTFormula) and
                formula.formula.decl().kind() == z3.Z3_OP_EQ and
                len(formula.free_variables()) == 2 and
                any(isinstance(fv, NonterminalPlaceholderVariable) for fv in formula.free_variables()) and
                any(isinstance(fv, StringPlaceholderVariable) for fv in formula.free_variables()) and
                all(is_z3_var(child) for child in formula.formula.children()))

    def query(self, formula: language.SMTFormula, var_map: Dict[PlaceholderVariable, str]) -> str:
        assert self.responsible(formula)

        free_vars: List[PlaceholderVariable] = cast(
            List[PlaceholderVariable], list(formula.free_variables()))

        npv = next(v for v in free_vars if isinstance(v, NonterminalPlaceholderVariable))
        spv = next(v for v in free_vars if isinstance(v, StringPlaceholderVariable))

        return f"eqstr({var_map[npv]}, {var_map[spv]})"

    def facts(self, tree: language.DerivationTree) -> Set[str]:
        result: Set[str] = set()

        for path, subtree in tree.paths():
            # We aim to ignore substrings. Some can be eliminated by ignoring nodes
            # which are children of a parent with the same type.
            parents = [tree.get_subtree(path[:idx]) for idx in range(len(path))]
            if any(parent.value == subtree.value for parent in parents):
                continue

            result.add(f'eqstr({subtree.id}, "' + str(subtree).replace('"', '""') + '")')

        return result

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, VariableEqualsStringTranslator)


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

    def facts(self, tree: language.DerivationTree) -> Set[str]:
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

        return set(result)

    def __hash__(self):
        return hash(type(self).__name__)

    def __eq__(self, other):
        return isinstance(other, CountPredicateTranslator)


def get_translator(formula: language.Formula, grammar: Grammar) -> Optional[PrologTranslator]:
    translators = [
        BeforePredicateTranslator(),
        CountPredicateTranslator(grammar),
        VariablesEqualTranslator(),
        VariableEqualsStringTranslator(),
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
