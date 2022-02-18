import functools
import io
import itertools
import logging
import os.path
import pkgutil
import re
from abc import ABC
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable, Sequence, Generator

import isla.fuzzer
import z3
from fuzzingbook.Parser import canonical
from grammar_graph import gg
from isla import language, isla_predicates
from isla.evaluator import evaluate
from isla.existential_helpers import paths_between
from isla.helpers import is_z3_var, z3_subst, dict_of_lists_to_list_of_dicts, RE_NONTERMINAL, weighted_geometric_mean, \
    visit_z3_expr
from isla.isla_predicates import reachable, is_before
from isla.language import set_smt_auto_eval
from isla.solver import ISLaSolver
from isla.type_defs import Grammar, ParseTree
from orderedset import OrderedSet
from pathos import multiprocessing as pmp

from islearn.helpers import parallel_all, connected_chains, replace_formula_by_formulas, transitive_closure, tree_in, \
    is_int
from islearn.language import NonterminalPlaceholderVariable, PlaceholderVariable, \
    NonterminalStringPlaceholderVariable, parse_abstract_isla, StringPlaceholderVariable, \
    AbstractISLaUnparser, MexprPlaceholderVariable, AbstractBindExpression
from islearn.mutation import MutationFuzzer
from islearn.parse_tree_utils import replace_path, filter_tree, tree_to_string, expand_tree, tree_leaves, \
    get_subtree

STANDARD_PATTERNS_REPO = "patterns.yaml"
logger = logging.getLogger("learner")


class InvariantLearner:
    def __init__(
            self,
            grammar: Grammar,
            prop: Callable[[language.DerivationTree], bool],
            positive_examples: Optional[Iterable[language.DerivationTree]] = None,
            negative_examples: Optional[Iterable[language.DerivationTree]] = None,
            patterns: Optional[List[language.Formula | str]] = None,
            pattern_file: Optional[str] = None,
            activated_patterns: Optional[Iterable[str]] = None,
            deactivated_patterns: Optional[Iterable[str]] = None,
            k: int = 3,
            target_number_positive_samples: int = 10,
            target_number_negative_samples: int = 10,
            target_number_positive_samples_for_learning: int = 10,
            filters: Sequence[str] = (
                    "Structural Predicates Filter",
                    "Variable Equality Filter",
                    "Nonterminal String in `count` Predicates Filter",
                    "String-to-Int Filter",
            ),
            mexpr_expansion_limit: int = 5):
        self.grammar = grammar
        self.canonical_grammar = canonical(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.prop = prop
        self.k = k
        self.filters = filters
        self.mexpr_expansion_limit = mexpr_expansion_limit

        self.positive_examples: List[language.DerivationTree] = list(set(positive_examples or []))
        self.original_positive_examples: List[language.DerivationTree] = list(self.positive_examples)
        self.negative_examples: List[language.DerivationTree] = list(set(negative_examples or []))

        self.target_number_positive_samples = target_number_positive_samples
        self.target_number_negative_samples = target_number_negative_samples
        self.target_number_positive_samples_for_learning = target_number_positive_samples_for_learning
        assert target_number_positive_samples >= target_number_positive_samples_for_learning

        assert all(prop(example) for example in self.positive_examples)
        assert all(not prop(example) for example in self.negative_examples)

        # Also consider inverted patterns?
        assert not activated_patterns or not deactivated_patterns
        if not patterns:
            pattern_repo = patterns_from_file(pattern_file or STANDARD_PATTERNS_REPO)
            if activated_patterns:
                self.patterns = [pattern for name in activated_patterns for pattern in pattern_repo[name]]
            else:
                self.patterns = list(pattern_repo.get_all(but=deactivated_patterns or []))
        else:
            self.patterns = [
                pattern if isinstance(pattern, language.Formula)
                else parse_abstract_isla(pattern, grammar)
                for pattern in patterns]

    def learn_invariants(self) -> Dict[language.Formula, float]:
        self._generate_more_inputs()
        assert len(self.positive_examples) > 0, "Cannot learn without any positive examples!"

        candidates = self.generate_candidates(self.patterns, self.positive_examples_for_learning)
        logger.info("Found %d invariant candidates.", len(candidates))

        # logger.debug(
        #     "Candidates:\n%s",
        #     "\n\n".join([language.ISLaUnparser(candidate).unparse() for candidate in candidates]))

        logger.info("Filtering invariants.")

        # Only consider *real* invariants

        # invariants = list(candidates)
        # test_inputs = list(self.positive_examples)
        # while invariants and test_inputs:
        #     inp = test_inputs.pop(0)
        #     with pmp.ProcessingPool(processes=2 * pmp.cpu_count()) as pool:
        #         invariants = [inv for inv in pool.map(
        #             lambda inv: (inv if evaluate(inv, inp, self.grammar).is_true() else None),
        #             invariants,
        #             chunksize=10
        #         ) if inv is not None]

        invariants = [
            inv for inv in candidates
            if all(evaluate(inv, inp, self.grammar).is_true() for inp in self.positive_examples)
        ]

        logger.info("%d invariants remain after filtering.", len(invariants))

        # ne_before = len(negative_examples)
        # negative_examples.update(generate_counter_examples_from_formulas(grammar, prop, invariants))
        # logger.info(
        #     "Generated %d additional negative samples (from invariants).",
        #     len(negative_examples) - ne_before
        # )

        logger.info("Calculating precision.")

        with pmp.ProcessingPool(processes=2 * pmp.cpu_count()) as pool:
            eval_results = pool.map(
                lambda t: (t[0], int(evaluate(t[0], t[1], self.grammar).is_true())),
                itertools.product(invariants, self.negative_examples),
                chunksize=10
            )

        result: Dict[language.Formula, float] = {
            inv: 1 - (sum([eval_result for other_inv, eval_result in eval_results if other_inv == inv])
                      / len(self.negative_examples))
            for inv in invariants
        }

        logger.info("Found %d invariants with non-zero precision.", len([p for p in result.values() if p > 0]))

        return dict(cast(List[Tuple[language.Formula, float]],
                         sorted(result.items(), key=lambda p: p[1], reverse=True)))

    def generate_candidates(
            self,
            patterns: Iterable[language.Formula | str],
            inputs: Iterable[language.DerivationTree]) -> Set[language.Formula]:
        input_reachability_relation = self._create_input_reachability_relation(inputs)

        logger.debug("Computed input reachability relation of size %d", len(input_reachability_relation))

        filters: List[PatternInstantiationFilter] = [
            StructuralPredicatesFilter(),
            VariablesEqualFilter(),
            NonterminalStringInCountPredicatesFilter(self.graph, input_reachability_relation),
            StringToIntFilter(),
        ]

        filters = [filter for filter in filters if filter.name in self.filters]

        patterns = [
            parse_abstract_isla(pattern, self.grammar) if isinstance(pattern, str)
            else pattern
            for pattern in patterns]

        result: Set[language.Formula] = set([])

        for pattern in patterns:
            logger.debug("Instantiating pattern\n%s", AbstractISLaUnparser(pattern).unparse())
            set_smt_auto_eval(pattern, False)

            # Instantiate various placeholder variables:
            # 1. Nonterminal placeholders
            pattern_insts_without_nonterminal_placeholders = self._instantiate_nonterminal_placeholders(
                pattern, input_reachability_relation)

            logger.debug("Found %d instantiations of pattern meeting quantifier requirements",
                         len(pattern_insts_without_nonterminal_placeholders))

            # 2. Match expression placeholders
            pattern_insts_without_mexpr_placeholders = self._instantiate_mexpr_placeholders(
                pattern_insts_without_nonterminal_placeholders)

            logger.debug("Found %d instantiations of pattern after instantiating match expression placeholders",
                         len(pattern_insts_without_mexpr_placeholders))

            # 3. Nonterminal-String placeholders
            pattern_insts_without_nonterminal_string_placeholders = self._instantiate_nonterminal_string_placeholders(
                pattern_insts_without_mexpr_placeholders)

            logger.debug("Found %d instantiations of pattern after instantiating nonterminal string placeholders",
                         len(pattern_insts_without_nonterminal_string_placeholders))

            # 4. String placeholders
            pattern_insts_without_string_placeholders = self._instantiate_string_placeholders(
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

    def _create_input_reachability_relation(self, inputs: Iterable[language.DerivationTree]) -> Set[Tuple[str, str]]:
        nonterminal_paths: Set[Tuple[str, ...]] = {
            tuple([inp.get_subtree(path[:idx]).value
                   for idx in range(len(path))])
            for inp in inputs
            for path, _ in inp.leaves()
        }

        input_reachability_relation: Set[Tuple[str, str]] = transitive_closure({
            pair
            for path in nonterminal_paths
            for pair in [(path[i], path[k]) for i in range(len(path)) for k in range(i + 1, len(path))]
        })

        logger.debug("Computed input reachability relation of size %d", len(input_reachability_relation))

        return input_reachability_relation

    def _generate_more_inputs(self):
        logger.info(
            "Starting with %d positive, and %d negative samples.",
            len(self.positive_examples),
            len(self.negative_examples)
        )

        if not self.positive_examples:
            ne_before = len(self.negative_examples)
            pe_before = len(self.positive_examples)
            self._generate_sample_inputs()

            logger.info(
                "Generated %d additional positive, and %d additional negative samples (by grammar fuzzing).",
                len(self.positive_examples) - pe_before,
                len(self.negative_examples) - ne_before
            )

        # logger.debug(
        #     "Positive examples:\n%s",
        #     "\n".join(map(str, self.positive_examples)))

        assert len(self.positive_examples) > 0, "Cannot learn without any positive examples!"
        pe_before = len(self.positive_examples)
        ne_before = len(self.negative_examples)
        mutation_fuzzer = MutationFuzzer(self.grammar, self.positive_examples, self.prop, k=self.k)
        for inp in mutation_fuzzer.run(num_iterations=50, alpha=.1, yield_negative=True):
            if self.prop(inp) and not tree_in(inp, self.positive_examples):
                self.positive_examples.append(inp)
            elif not self.prop(inp) and not tree_in(inp, self.negative_examples):
                self.negative_examples.append(inp)

        # logger.debug(
        #     "Positive examples:\n%s",
        #     "\n".join(map(str, self.positive_examples)))

        logger.info(
            "Generated %d additional positive, and %d additional negative samples (by mutation fuzzing).",
            len(self.positive_examples) - pe_before,
            len(self.negative_examples) - ne_before
        )

        if (len(self.positive_examples) < self.target_number_positive_samples or
                len(self.negative_examples) < self.target_number_negative_samples):
            ne_before = len(self.negative_examples)
            pe_before = len(self.positive_examples)
            self._generate_sample_inputs()

            # logger.debug(
            #     "Positive examples:\n%s",
            #     "\n".join(map(str, self.positive_examples)))

            logger.info(
                "Generated %d additional positive, and %d additional negative samples (by grammar fuzzing).",
                len(self.positive_examples) - pe_before,
                len(self.negative_examples) - ne_before
            )

        self.positive_examples = self.sort_inputs(
            self.positive_examples,
            more_paths_weight=1.5,
            smaller_inputs_weight=1.0,
        )[:self.target_number_positive_samples]

        self.negative_examples = \
            self.sort_inputs(
                self.negative_examples,
                more_paths_weight=2.0,
                smaller_inputs_weight=1.0
            )[:self.target_number_negative_samples]

        self.positive_examples_for_learning = \
            self.sort_inputs(
                self.positive_examples,
                more_paths_weight=1.7,
                smaller_inputs_weight=1.0)[:self.target_number_positive_samples_for_learning]

        logger.info(
            "Reduced positive / negative samples to subsets of %d / %d samples based on k-path coverage, "
            "keeping %d positive examples for candidate generation.",
            len(self.positive_examples),
            len(self.negative_examples),
            len(self.positive_examples_for_learning),
        )

        logger.debug(
            "Positive examples:\n%s",
            "\n".join(map(str, self.positive_examples)))

        logger.debug(
            "Examples for learning:\n%s",
            "\n".join(map(str, self.positive_examples_for_learning)))

    def _generate_sample_inputs(
            self,
            num_tries: int = 100) -> None:
        fuzzer = isla.fuzzer.GrammarCoverageFuzzer(self.grammar)
        if (len(self.positive_examples) < self.target_number_positive_samples or
                len(self.negative_examples) < self.target_number_negative_samples):
            i = 0
            while ((len(self.positive_examples) < self.target_number_positive_samples
                    or len(self.negative_examples) < self.target_number_negative_samples)
                   and i < num_tries):
                i += 1

                inp = fuzzer.expand_tree(language.DerivationTree("<start>", None))

                if self.prop(inp) and not tree_in(inp, self.positive_examples):
                    self.positive_examples.append(inp)
                elif not self.prop(inp) and not tree_in(inp, self.negative_examples):
                    self.negative_examples.append(inp)

    def _generate_counter_examples_from_formulas(
            self,
            formulas: Iterable[language.Formula],
            desired_number_counter_examples: int = 50,
            num_tries: int = 100) -> Set[language.DerivationTree]:
        result: Set[language.DerivationTree] = set()

        solvers = {
            formula: ISLaSolver(self.grammar, formula, enforce_unique_trees_in_queue=False).solve()
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
                    if not self.prop(inp):
                        result.add(inp)

            i += 1

        return result

    def sort_inputs(
            self,
            inputs: Iterable[language.DerivationTree],
            more_paths_weight: float = 1.0,
            smaller_inputs_weight: float = 0.0) -> List[language.DerivationTree]:
        assert more_paths_weight or smaller_inputs_weight
        inputs = set(inputs)
        result: List[language.DerivationTree] = []

        tree_paths = {inp: self.graph.k_paths_in_tree(inp.to_parse_tree(), self.k) for inp in inputs}
        covered_paths: Set[Tuple[gg.Node, ...]] = set([])
        max_len_input = max(len(inp) for inp in inputs)

        def uncovered_paths(inp: language.DerivationTree) -> Set[Tuple[gg.Node, ...]]:
            return {path for path in tree_paths[inp] if path not in covered_paths}

        def sort_by_paths_key(inp: language.DerivationTree) -> float:
            return len(uncovered_paths(inp))

        def sort_by_length_key(inp: language.DerivationTree) -> float:
            return len(inp)

        def sort_by_paths_and_length_key(inp: language.DerivationTree) -> float:
            return weighted_geometric_mean(
                [len(uncovered_paths(inp)), max_len_input - len(inp)],
                [more_paths_weight, smaller_inputs_weight])

        if not more_paths_weight:
            key = sort_by_length_key
        elif not smaller_inputs_weight:
            key = sort_by_paths_key
        else:
            key = sort_by_paths_and_length_key

        while inputs:
            inp = sorted(inputs, key=key, reverse=True)[0]

            result.append(inp)
            inputs.remove(inp)

        return result

    @staticmethod
    def _instantiate_nonterminal_placeholders(
            pattern: language.Formula,
            input_reachability_relation: Set[Tuple[str, str]]) -> Set[language.Formula]:
        in_visitor = InVisitor()
        pattern.accept(in_visitor)
        variable_chains: Set[Tuple[language.Variable, ...]] = connected_chains(in_visitor.result)
        assert all(chain[-1] == extract_top_level_constant(pattern) for chain in variable_chains)

        instantiations: List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]] = []
        for variable_chain in variable_chains:
            nonterminal_sequences: Set[Tuple[str, ...]] = set([])

            partial_sequences: Set[Tuple[str, ...]] = {("<start>",)}
            while partial_sequences:
                partial_sequence = next(iter(partial_sequences))
                partial_sequences.remove(partial_sequence)
                if len(partial_sequence) == len(variable_chain):
                    nonterminal_sequences.add(tuple(reversed(partial_sequence)))
                    continue

                partial_sequences.update({
                    partial_sequence + (to_nonterminal,)
                    for (from_nonterminal, to_nonterminal) in input_reachability_relation
                    if from_nonterminal == partial_sequence[-1]})

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

        result: Set[language.Formula] = {
            pattern.substitute_variables(instantiation)
            for instantiation in instantiations
        }

        return result

    def _instantiate_mexpr_placeholders(
            self,
            inst_patterns: Set[language.Formula]) -> Set[language.Formula]:
        def quantified_formulas_with_mexpr_phs(formula: language.Formula) -> Set[language.QuantifiedFormula]:
            return cast(Set[language.QuantifiedFormula], set(language.FilterVisitor(
                lambda f: (isinstance(f, language.QuantifiedFormula) and
                           f.bind_expression is not None and
                           isinstance(f.bind_expression, AbstractBindExpression) and
                           isinstance(f.bind_expression.bound_elements[0], MexprPlaceholderVariable))).collect(
                formula)))

        if not any(quantified_formulas_with_mexpr_phs(pattern) for pattern in inst_patterns):
            return inst_patterns

        result: Set[language.Formula] = set([])
        stack: List[language.Formula] = list(inst_patterns)
        while stack:
            pattern = stack.pop()
            qfd_formulas_with_mexpr_phs = quantified_formulas_with_mexpr_phs(pattern)
            if not qfd_formulas_with_mexpr_phs:
                result.add(pattern)
                continue

            for qfd_formula_w_mexpr_phs in qfd_formulas_with_mexpr_phs:
                mexpr_ph: MexprPlaceholderVariable = cast(
                    AbstractBindExpression,
                    qfd_formula_w_mexpr_phs.bind_expression).bound_elements[0]
                in_nonterminal = qfd_formula_w_mexpr_phs.bound_variable.n_type
                nonterminal_types = tuple([var.n_type for var in mexpr_ph.variables])

                for mexpr_str in self._infer_mexpr(in_nonterminal, nonterminal_types):
                    def replace_with_var(elem: str) -> str | language.Variable:
                        try:
                            return next(var for idx, var in enumerate(mexpr_ph.variables)
                                        if elem == var.n_type.replace(">", f"-{hash((var.n_type, idx))}>"))
                        except StopIteration:
                            return elem

                    mexpr_elements = [
                        replace_with_var(token)
                        for token in
                        re.split(RE_NONTERMINAL, mexpr_str)
                        if token]

                    constructor = (
                        language.ForallFormula if isinstance(qfd_formula_w_mexpr_phs, language.ForallFormula)
                        else language.ExistsFormula)

                    new_formula = constructor(
                        qfd_formula_w_mexpr_phs.bound_variable,
                        qfd_formula_w_mexpr_phs.in_variable,
                        qfd_formula_w_mexpr_phs.inner_formula,
                        language.BindExpression(*mexpr_elements))

                    stack.append(language.replace_formula(pattern, qfd_formula_w_mexpr_phs, new_formula))

        return result

    @lru_cache(maxsize=None)
    def _infer_mexpr(
            self,
            in_nonterminal: str,
            nonterminal_types: Tuple[str]) -> Set[str]:
        result: Set[str] = set([])

        candidate_trees: List[ParseTree] = [(in_nonterminal, None)]
        i = 0
        while candidate_trees and i <= self.mexpr_expansion_limit:
            i += 1
            tree = candidate_trees.pop(0)

            # If the candidate tree has the shape
            #
            #          qfd_nonterminal
            #                 |
            #         other_nonterminal
            #                 |
            #          ... subtree ...
            #
            # then we could as well quantify over `other_nonterminal`. Consequently,
            # we prune this candidate. The other option is very likely also among
            # inst_patterns.

            if (tree[1] is not None and
                    len(tree[1]) == 1 and
                    tree[1][0][1]):
                continue

            nonterminal_occurrences = [
                [path for path, _ in filter_tree(tree, lambda t: t[0] == ntype)]
                for ntype in nonterminal_types]

            if all(occ for occ in nonterminal_occurrences):
                product = list(itertools.product(*nonterminal_occurrences))
                matching_seqs = [
                    seq for seq in product
                    if (list(seq) == sorted(seq) and
                        all(is_before(None, seq[idx], seq[idx + 1]) for idx in range(len(seq) - 1)))]

                for matching_seq in matching_seqs:
                    assert len(matching_seq) == len(nonterminal_types)

                    # We change node labels to be able to correctly identify variable positions in the tree.
                    bind_expr_tree = tree
                    for idx, path in enumerate(matching_seq):
                        ntype = nonterminal_types[idx]
                        bind_expr_tree = replace_path(
                            bind_expr_tree,
                            path,
                            (ntype.replace(">", f"-{hash((ntype, idx))}>"), None))

                    # We prune too specific leaves of the tree: Each node that does not contain
                    # a variable node as a child, and is an immediate child of a node containing
                    # a variable node, is pruned away.
                    non_var_paths = [path for path, _ in tree_leaves(bind_expr_tree)
                                     if path not in matching_seq]

                    for path in non_var_paths:
                        if len(path) < 2:
                            continue

                        while len(path) > 1 and not any(
                                len(path[:-1]) < len(var_path) and
                                var_path[:len(path[:-1])] == path[:-1]
                                for var_path in matching_seq):
                            path = path[:-1]

                        if len(path) > 0 and not any(
                                len(path) < len(var_path) and
                                var_path[:len(path)] == path
                                for var_path in matching_seq):
                            bind_expr_tree = replace_path(
                                bind_expr_tree,
                                path,
                                (get_subtree(bind_expr_tree, path)[0], None))

                    result.add(tree_to_string(bind_expr_tree, show_open_leaves=True))

            candidate_trees.extend(expand_tree(tree, self.canonical_grammar))

        return result

    def _instantiate_nonterminal_string_placeholders(
            self,
            inst_patterns: Set[language.Formula]) -> Set[language.Formula]:
        if all(not isinstance(placeholder, NonterminalStringPlaceholderVariable)
               for inst_pattern in inst_patterns
               for placeholder in get_placeholders(inst_pattern)):
            return inst_patterns

        result: Set[language.Formula] = set([])
        for inst_pattern in inst_patterns:
            for nonterminal in self.grammar:
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

    def _instantiate_string_placeholders(
            self,
            inst_patterns: Set[language.Formula],
            inputs: Iterable[language.DerivationTree]) -> Set[language.Formula]:
        if all(not isinstance(placeholder, StringPlaceholderVariable)
               for inst_pattern in inst_patterns
               for placeholder in get_placeholders(inst_pattern)):
            return inst_patterns

        fragments: Dict[str, Set[str]] = {
            nonterminal: functools.reduce(
                set.intersection,
                [
                    {
                        str(subtree)
                        for _, subtree in inp.filter(lambda t: t.value == nonterminal)
                        if str(subtree)
                    }
                    for inp in inputs
                ]
            )
            for nonterminal in self.grammar
        }

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

                insts = functools.reduce(set.__or__, [fragments.get(var.n_type, set([])) for var in non_ph_vars])

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


def get_quantifier_block(formula: language.Formula) -> List[language.QuantifiedFormula]:
    if isinstance(formula, language.QuantifiedFormula):
        return [formula] + get_quantifier_block(formula.inner_formula)

    if isinstance(formula, language.NumericQuantifiedFormula):
        return get_quantifier_block(formula.inner_formula)

    return []


def nonterminal_chain_closure(
        chain: Tuple[str, ...],
        graph: gg.GrammarGraph,
        max_num_cycles=0) -> Set[Tuple[str, ...]]:
    closure: Set[Tuple[str, ...]] = {(chain[0],)}
    for chain_elem in chain[1:]:
        old_chain_1_closure = set(closure)
        closure = set([])
        for partial_chain in old_chain_1_closure:
            closure.update({
                partial_chain + path[1:]
                for path in paths_between(graph, partial_chain[-1], chain_elem)})

    return closure


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

                if free_vars[0].n_type != free_vars[1].n_type:
                    # NOTE: In all existing case studies, we are only stipulating equality
                    #       of variables with equal nonterminal type. It might be possible
                    #       to do this with variables that are in a supertype relation, but
                    #       this gives us fewer chances of filtering.
                    success = False
                    break

                trees = inp.filter(lambda t: t.value == free_vars[0].n_type)

                if not any(
                        t1.value == free_vars[0].n_type and
                        t2.value == free_vars[1].n_type
                        for (p1, t1), (p2, t2)
                        in itertools.product(trees, trees)
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
        # TODO: This does not work properly when we have non-conjunctive cores! E.g., in a disjunction
        #       or inside a negation, this filter is too restrictive.

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
    def __init__(self, graph: gg.GrammarGraph, input_reachability_relation: Set[Tuple[str, str]]):
        super().__init__("Nonterminal String in `count` Predicates Filter")
        self.graph = graph
        self.input_reachability_relation = input_reachability_relation

    def reachable(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return reachable(self.graph, from_nonterminal, to_nonterminal)

    def reachable_in_inputs(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return (from_nonterminal, to_nonterminal) in self.input_reachability_relation

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


class StringToIntFilter(PatternInstantiationFilter):
    def __init__(self):
        super().__init__("String-to-Int Filter")

    def predicate(self, formula: language.Formula, inputs: Iterable[language.DerivationTree]) -> bool:
        str_to_int_expressions_with_vars: List[Tuple[Set[z3.ExprRef], OrderedSet[language.Variable]]] = [
            (
                {z3_expr for z3_expr in set(visit_z3_expr(cast(language.SMTFormula, smt_for).formula))
                 if z3_expr.decl().kind() == z3.Z3_OP_STR_TO_INT},
                smt_for.free_variables())
            for smt_for in language.FilterVisitor(lambda f: isinstance(f, language.SMTFormula)).collect(formula)
        ]

        str_to_int_expressions_with_vars = [p for p in str_to_int_expressions_with_vars if p[0]]

        if not str_to_int_expressions_with_vars:
            return True

        def get_child(expr: z3.ExprRef) -> z3.SeqRef:
            return expr.children()[0]

        if any(z3.is_string_value(get_child(str_to_int_expression)) and
               not is_int(get_child(str_to_int_expression).as_string())
               for str_to_int_expressions, free_vars in str_to_int_expressions_with_vars
               for str_to_int_expression in str_to_int_expressions):
            return False

        for inp in inputs:
            success = True
            for str_to_int_expressions, free_vars in str_to_int_expressions_with_vars:
                for str_to_int_expression in str_to_int_expressions:
                    if not is_z3_var(get_child(str_to_int_expression)):
                        continue

                    n_type = next(
                        var.n_type for var in free_vars
                        if var.name == get_child(str_to_int_expression).as_string())

                    trees = [t for _, t in inp.filter(lambda t: t.value == n_type)]
                    if not trees:
                        success = False
                        break

                    # Success if there is a tree with an Integer value.
                    if not any(is_int(str(tree)) for tree in trees):
                        success = False
                        break

                if not success:
                    break

                # if not any(structural_formula.substitute_expressions(arg_substitution).evaluate(inp)
                #            for arg_substitution in arg_substitutions):
                #     success = False
                #     break

            if success:
                return True

        return False


class InVisitor(language.FormulaVisitor):
    def __init__(self):
        self.result: Set[Tuple[language.Variable, language.Variable]] = set()

    def visit_exists_formula(self, formula: language.ExistsFormula):
        self.handle(formula)

    def visit_forall_formula(self, formula: language.ForallFormula):
        self.handle(formula)

    def handle(self, formula: language.QuantifiedFormula):
        self.result.add((formula.bound_variable, formula.in_variable))
        if (formula.bind_expression and
                any(isinstance(elem, MexprPlaceholderVariable)
                    for elem in formula.bind_expression.bound_elements)):
            phs = [elem for elem in formula.bind_expression.bound_elements
                   if isinstance(elem, MexprPlaceholderVariable)]
            assert len(phs) == 1
            ph = cast(MexprPlaceholderVariable, phs[0])
            for nonterminal_placeholder in ph.variables:
                self.result.add((nonterminal_placeholder, formula.bound_variable))


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


def extract_top_level_constant(candidate):
    return next(
        (c for c in language.VariablesCollector.collect(candidate)
         if isinstance(c, language.Constant) and not c.is_numeric()))


class PatternRepository:
    DEFAULT_GROUP = "default"

    def __init__(self, data: List[Dict[str, str]]):
        self.groups: Dict[str, Dict[str, language.Formula]] = {}
        for entry in data:
            name = entry["name"]
            group = entry.get("group", PatternRepository.DEFAULT_GROUP)
            constraint = parse_abstract_isla(entry["constraint"])
            self.groups.setdefault(group, {})[name] = constraint

    def __getitem__(self, item: str) -> Set[language.Formula]:
        if item in self.groups:
            return set(self.groups[item].values())

        for elements in self.groups.values():
            if item in elements:
                return {elements[item]}

        return set([])

    def __contains__(self, item: str) -> bool:
        return bool(self[item])

    def __len__(self):
        return sum([len(group.values()) for group in self.groups.values()])

    def get_all(self, but: Iterable[str] = tuple()) -> Set[language.Formula]:
        exclude: Set[language.Formula] = set([])
        for excluded in but:
            exclude.update(self[excluded])

        all_patterns: Set[language.Formula] = set(
            functools.reduce(set.__or__, [set(d.values()) for d in self.groups.values()]))

        return all_patterns.intersection(exclude)

    def __str__(self):
        result = ""
        for group in self.groups:
            for name in self.groups[group]:
                result += f"- name: {name}\n"
                if group != PatternRepository.DEFAULT_GROUP:
                    result += f"  group: {group}\n"
                result += "  constraint: |\n"
                constraint_str = AbstractISLaUnparser(self.groups[group][name]).unparse()
                constraint_str = "\n".join(["      " + line for line in constraint_str.split("\n")])
                result += constraint_str
                result += "\n\n"

        return result.strip()


def patterns_from_file(file_name: str = STANDARD_PATTERNS_REPO) -> PatternRepository:
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

    return PatternRepository(data)
