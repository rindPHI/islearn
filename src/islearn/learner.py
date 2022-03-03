import copy
import functools
import inspect
import itertools
import logging
import os.path
import pkgutil
import re
from abc import ABC
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable, Sequence

import isla.fuzzer
import toml
import z3
from fuzzingbook.Parser import canonical
from grammar_graph import gg
from isla import language, isla_predicates
from isla.evaluator import evaluate, matches_for_quantified_formula
from isla.helpers import RE_NONTERMINAL, weighted_geometric_mean, \
    is_nonterminal, dict_of_lists_to_list_of_dicts
from isla.isla_predicates import reachable, is_before
from isla.language import set_smt_auto_eval
from isla.solver import ISLaSolver
from isla.three_valued_truth import ThreeValuedTruth
from isla.type_defs import Grammar, ParseTree, Path
from isla.z3_helpers import z3_subst, evaluate_z3_expression, is_valid, \
    DomainError
from pathos import multiprocessing as pmp

from islearn.helpers import connected_chains, replace_formula_by_formulas, transitive_closure, tree_in, \
    is_int, is_float
from islearn.language import NonterminalPlaceholderVariable, PlaceholderVariable, \
    NonterminalStringPlaceholderVariable, parse_abstract_isla, StringPlaceholderVariable, \
    AbstractISLaUnparser, MexprPlaceholderVariable, AbstractBindExpression
from islearn.mutation import MutationFuzzer
from islearn.parse_tree_utils import replace_path, filter_tree, tree_to_string, expand_tree, tree_leaves, \
    get_subtree

STANDARD_PATTERNS_REPO = "patterns.toml"
logger = logging.getLogger("learner")


class TruthTableRow:
    def __init__(
            self,
            formula: language.Formula,
            inputs: Sequence[language.DerivationTree] = (),
            eval_results: Sequence[bool] = ()):
        self.formula = formula
        self.inputs = list(inputs)
        self.eval_results: List[bool] = list(eval_results)

    def evaluate(
            self,
            grammar: Grammar,
            columns_parallel: bool = False,
            lazy: bool = False,
            result_threshold: float = .9) -> 'TruthTableRow':
        """If lazy is True, then the evaluation stops as soon as result_threshold can no longer be
        reached. E.g., if result_threshold is .9 and there are 100 inputs, then after more than
        10 negative results, 90% positive results is no longer possible."""

        if columns_parallel:
            with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                iterator = pool.imap(
                    lambda inp: evaluate(self.formula, inp, grammar).is_true(),
                    self.inputs)

                self.eval_results = []
                negative_results = 0
                for eval_result in iterator:
                    if lazy and negative_results > len(self.inputs) * (1 - result_threshold):
                        self.eval_results += [False for _ in range(len(self.inputs) - len(self.eval_results))]
                        break

                    if not eval_result:
                        negative_results += 1
                    self.eval_results.append(eval_result)
        else:
            self.eval_results = []
            negative_results = 0
            for inp in self.inputs:
                if lazy and negative_results > len(self.inputs) * (1 - result_threshold):
                    self.eval_results += [False for _ in range(len(self.inputs) - len(self.eval_results))]
                    break

                eval_result = evaluate(self.formula, inp, grammar).is_true()
                if not eval_result:
                    negative_results += 1
                self.eval_results.append(eval_result)

        return self

    def eval_result(self) -> float:
        assert len(self.inputs) > 0
        assert len(self.eval_results) == len(self.inputs)
        assert all(isinstance(entry, bool) for entry in self.eval_results)
        return sum(int(entry) for entry in self.eval_results) / len(self.eval_results)

    def __repr__(self):
        return f"TruthTableRow({repr(self.formula)}, {repr(self.inputs)}, {repr(self.eval_results)})"

    def __str__(self):
        return f"{self.formula}: {', '.join(map(str, self.eval_results))}"

    def __eq__(self, other):
        return (isinstance(other, TruthTableRow) and
                self.formula == other.formula)

    def __hash__(self):
        return hash(self.formula)

    def __neg__(self):
        return TruthTableRow(
            -self.formula,
            self.inputs,
            [not eval_result for eval_result in self.eval_results]
        )

    def __and__(self, other: 'TruthTableRow') -> 'TruthTableRow':
        assert len(self.inputs) == len(other.inputs)
        assert len(self.eval_results) == len(other.eval_results)
        return TruthTableRow(
            self.formula & other.formula,
            self.inputs,
            [a and b for a, b in zip(self.eval_results, other.eval_results)]
        )

    def __or__(self, other: 'TruthTableRow') -> 'TruthTableRow':
        assert len(self.inputs) == len(other.inputs)
        assert len(self.eval_results) == len(other.eval_results)
        return TruthTableRow(
            self.formula | other.formula,
            self.inputs,
            [a or b for a, b in zip(self.eval_results, other.eval_results)]
        )


class TruthTable:
    def __init__(self, rows: Iterable[TruthTableRow] = ()):
        self.rows = set(rows)

    def __repr__(self):
        return f"TruthTable({repr(self.rows)})"

    def __str__(self):
        return "\n".join(map(str, self.rows))

    def __iter__(self):
        return iter(self.rows)

    def __add__(self, other: 'TruthTable') -> 'TruthTable':
        return TruthTable(self.rows | other.rows)

    def __iadd__(self, other: 'TruthTable') -> 'TruthTable':
        self.rows |= other.rows
        return self

    def __or__(self, other: Iterable[TruthTableRow]) -> 'TruthTable':
        return TruthTable(self.rows | set(other))

    def __ior__(self, other: Iterable[TruthTableRow]) -> 'TruthTable':
        self.rows |= set(other)
        return self

    def evaluate(
            self,
            grammar: Grammar,
            columns_parallel: bool = False,
            rows_parallel: bool = False,
            lazy: bool = False,
            result_threshold: float = .9) -> 'TruthTable':
        """If lazy is True, then column evaluation stops as soon as result_threshold can no longer be
        reached. E.g., if result_threshold is .9 and there are 100 inputs, then after more than
        10 negative results, 90% positive results is no longer possible."""

        assert not columns_parallel or not rows_parallel

        if rows_parallel:
            with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                self.rows = set(pool.map(
                    lambda row: row.evaluate(grammar, columns_parallel, lazy=lazy, result_threshold=result_threshold),
                    self.rows
                ))
        else:
            for row in self.rows:
                row.evaluate(grammar, columns_parallel, lazy=lazy, result_threshold=result_threshold)

        return self


class InvariantLearner:
    def __init__(
            self,
            grammar: Grammar,
            prop: Optional[Callable[[language.DerivationTree], bool]] = None,
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
            mexpr_expansion_limit: int = 5,
            min_recall: float = .9,
            min_precision: float = .6,
            max_disjunction_size: int = 1,
            max_conjunction_size: int = 2):
        # We add extended caching certain, crucial functions.
        isla.helpers.evaluate_z3_expression = lru_cache(maxsize=None)(
            inspect.unwrap(evaluate_z3_expression))
        isla.language.DerivationTree.__str__ = lru_cache(maxsize=None)(
            inspect.unwrap(isla.language.DerivationTree.__str__))
        isla.language.DerivationTree.paths = lru_cache(maxsize=128)(
            inspect.unwrap(isla.language.DerivationTree.paths))
        isla.language.DerivationTree.__hash__ = lambda tree: tree.id
        isla.isla_predicates.is_nth = lru_cache(maxsize=128)(
            inspect.unwrap(isla.isla_predicates.is_nth))

        self.grammar = grammar
        self.canonical_grammar = canonical(grammar)
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.prop = prop
        self.k = k
        self.mexpr_expansion_limit = mexpr_expansion_limit
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_disjunction_size = max_disjunction_size
        self.max_conjunction_size = max_conjunction_size

        self.positive_examples: List[language.DerivationTree] = list(set(positive_examples or []))
        self.original_positive_examples: List[language.DerivationTree] = list(self.positive_examples)
        self.negative_examples: List[language.DerivationTree] = list(set(negative_examples or []))
        self.positive_examples_for_learning: List[language.DerivationTree] = []

        self.target_number_positive_samples = target_number_positive_samples
        self.target_number_negative_samples = target_number_negative_samples
        self.target_number_positive_samples_for_learning = target_number_positive_samples_for_learning
        assert target_number_positive_samples >= target_number_positive_samples_for_learning

        assert not prop or all(prop(example) for example in self.positive_examples)
        assert not prop or all(not prop(example) for example in self.negative_examples)

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
        if self.prop:
            self._generate_more_inputs()
            assert len(self.positive_examples) > 0, "Cannot learn without any positive examples!"
            assert all(self.prop(positive_example) for positive_example in self.positive_examples)
            assert all(not self.prop(negative_example) for negative_example in self.negative_examples)

        self.positive_examples_for_learning = \
            self.sort_inputs(
                self.positive_examples,
                more_paths_weight=1.7,
                smaller_inputs_weight=1.0)[:self.target_number_positive_samples_for_learning]

        logger.info(
            "Keeping %d positive examples for candidate generation.",
            len(self.positive_examples_for_learning)
        )

        logger.debug(
            "Examples for learning:\n%s",
            "\n".join(map(str, self.positive_examples_for_learning)))

        candidates = self.generate_candidates(self.patterns, self.positive_examples_for_learning)
        logger.info("Found %d invariant candidates.", len(candidates))

        # logger.debug(
        #     "Candidates:\n%s",
        #     "\n\n".join([language.ISLaUnparser(candidate).unparse() for candidate in candidates]))

        logger.info("Filtering invariants.")

        # Only consider *real* invariants

        # NOTE: Disabled parallel evaluation for now. In certain cases, this renders
        #       the filtering process *much* slower, or gives rise to stack overflows
        #       (e.g., "test_learn_from_islearn_patterns_file" example).
        recall_truth_table = TruthTable([
            TruthTableRow(inv, self.positive_examples)
            for inv in candidates
        ]).evaluate(
            self.grammar,
            # rows_parallel=True,
            lazy=self.max_disjunction_size < 2,
            result_threshold=self.min_recall)

        if self.max_disjunction_size > 1:
            logger.info("Calculating recall of Boolean combinations.")

            disjunctive_precision_truthtable = copy.copy(recall_truth_table)
            # TODO: Find a way to deal with negations that does not induce many spurious invariants.
            #       Problem: Negation might have bad recall, that is improved by building a disjunction...
            #       Also, in the ALHAZEN-SQRT example, this gave rise to negated invs with meaningless
            #       constants, so we might have to control this somehow, at least with a config param.
            # disjunctive_precision_truthtable |= {-row for row in recall_truth_table}
            for level in range(2, self.max_disjunction_size + 1):
                logger.debug(f"Disjunction size: {level}")
                for rows in itertools.product(*[recall_truth_table for _ in range(level)]):
                    if not all(row_1 != row_2
                               for idx_1, row_1 in enumerate(rows)
                               for idx_2, row_2 in enumerate(rows)
                               if idx_1 != idx_2):
                        continue

                    disjunction = functools.reduce(TruthTableRow.__or__, rows)
                    new_eval_result = disjunction.eval_result()
                    if not all(new_eval_result > row.eval_result() for row in rows):
                        continue

                    disjunctive_precision_truthtable.rows.add(disjunction)

            recall_truth_table = disjunctive_precision_truthtable

        # TODO: Prefer stronger invariants, if any: Formulas with *the same* quantifier blocks imply each
        #       other if the qfr-free cores imply each other. If a stronger inv has the same recall than
        #       a weaker one, drop the weaker one. This is basically a static precision filter.

        invariants = {
            language.ensure_unique_bound_variables(row.formula) for row in recall_truth_table
            if row.eval_result() >= self.min_recall
        }

        logger.info("%d invariants remain after filtering.", len(invariants))
        # logger.debug("Invariants:\n%s", "\n\n".join(map(lambda f: language.ISLaUnparser(f).unparse(), invariants)))

        # ne_before = len(negative_examples)
        # negative_examples.update(generate_counter_examples_from_formulas(grammar, prop, invariants))
        # logger.info(
        #     "Generated %d additional negative samples (from invariants).",
        #     len(negative_examples) - ne_before
        # )

        if not self.negative_examples:
            invariants = sorted(list(invariants), key=lambda inv: len(inv))
            return {inv: 1.0 for inv in invariants}

        logger.info("Evaluating precision.")
        # logger.debug("Negative samples:\n" + "\n-----------\n".join(map(str, self.negative_examples)))

        precision_truth_table = TruthTable([
            TruthTableRow(inv, self.negative_examples)
            for inv in invariants
        ]).evaluate(
            self.grammar,
            # rows_parallel=True,
        )

        logger.info("Calculating precision of Boolean combinations.")

        conjunctive_precision_truthtable = copy.copy(precision_truth_table)
        for level in range(2, self.max_conjunction_size + 1):
            logger.debug(f"Conjunction size: {level}")
            for rows in itertools.product(*[precision_truth_table for _ in range(level)]):
                if not all(row_1 != row_2
                           for idx_1, row_1 in enumerate(rows)
                           for idx_2, row_2 in enumerate(rows)
                           if idx_1 != idx_2):
                    continue

                disjunction = functools.reduce(TruthTableRow.__and__, rows)
                new_eval_result = disjunction.eval_result()
                if not all(new_eval_result < row.eval_result() for row in rows):
                    continue

                conjunctive_precision_truthtable.rows.add(disjunction)

        precision_truth_table = conjunctive_precision_truthtable

        result: Dict[language.Formula, float] = {
            language.ensure_unique_bound_variables(row.formula): 1 - row.eval_result()
            for row in precision_truth_table
            if 1 - row.eval_result() >= self.min_precision
        }

        logger.info("Found %d invariants with non-zero precision.", len([p for p in result.values() if p > 0]))

        return dict(
            cast(List[Tuple[language.Formula, float]],
                 sorted(result.items(),
                        key=lambda p: (p[1], -len(p[0])),
                        reverse=True)))

    def generate_candidates(
            self,
            patterns: Iterable[language.Formula | str],
            inputs: Iterable[language.DerivationTree]) -> Set[language.Formula]:
        input_reachability_relation = create_input_reachability_relation(inputs)

        logger.debug("Computed input reachability relation of size %d", len(input_reachability_relation))

        filters: List[PatternInstantiationFilter] = [
            NonterminalStringInCountPredicatesFilter(self.graph, input_reachability_relation),
        ]

        patterns = [
            parse_abstract_isla(pattern, self.grammar) if isinstance(pattern, str)
            else pattern
            for pattern in patterns]

        inputs_subtrees: List[Dict[Path, language.DerivationTree]] = [dict(inp.paths()) for inp in inputs]

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

            pattern_insts_without_mexpr_placeholders = self._filter_partial_instantiations(
                pattern_insts_without_mexpr_placeholders, inputs_subtrees)
            logger.debug("%d instantiations remain after filtering",
                         len(pattern_insts_without_mexpr_placeholders))

            # 3. Special string placeholders in predicates.
            #    This comprises, e.g., `nth(<STRING>, elem, container`.
            pattern_insts_without_special_string_placeholders = \
                self.__instantiate_special_predicate_string_placeholders(
                    pattern_insts_without_mexpr_placeholders, inputs_subtrees)

            logger.debug("Found %d instantiations of pattern after instantiating special predicate string placeholders",
                         len(pattern_insts_without_special_string_placeholders))

            pattern_insts_without_special_string_placeholders = self._filter_partial_instantiations(
                pattern_insts_without_special_string_placeholders, inputs_subtrees)
            logger.debug("%d instantiations remain after filtering",
                         len(pattern_insts_without_special_string_placeholders))

            # 4. Nonterminal-String placeholders
            pattern_insts_without_nonterminal_string_placeholders = self._instantiate_nonterminal_string_placeholders(
                pattern_insts_without_special_string_placeholders)

            logger.debug("Found %d instantiations of pattern after instantiating nonterminal string placeholders",
                         len(pattern_insts_without_nonterminal_string_placeholders))

            pattern_insts_without_nonterminal_string_placeholders = self._filter_partial_instantiations(
                pattern_insts_without_nonterminal_string_placeholders, inputs_subtrees)
            logger.debug("%d instantiations remain after filtering",
                         len(pattern_insts_without_nonterminal_string_placeholders))

            pattern_insts_without_nonterminal_string_placeholders = self._apply_filters(
                pattern_insts_without_nonterminal_string_placeholders, filters, 1, inputs_subtrees)

            # 5. String placeholders
            string_placeholder_insts = self._get_string_placeholder_instantiations(
                pattern_insts_without_nonterminal_string_placeholders, inputs_subtrees)

            pattern_insts_without_string_placeholders: Set[language.Formula] = set([])
            for formula in string_placeholder_insts:
                if not string_placeholder_insts[formula]:
                    pattern_insts_without_string_placeholders.add(formula)
                    continue

                instantiations: List[Dict[StringPlaceholderVariable, str]] = dict_of_lists_to_list_of_dicts(
                    string_placeholder_insts[formula])
                for instantiation in instantiations:
                    if any(not approximately_evaluate_abst_for(
                            formula,
                            self.grammar,
                            {language.Constant("start", "<start>"): ((), subtrees[()])} | instantiation,
                            subtrees).is_false() for subtrees in inputs_subtrees):
                        instantiated_formula = formula
                        for ph, inst in instantiation.items():
                            instantiated_formula = language.replace_formula(
                                instantiated_formula,
                                functools.partial(substitute_string_placeholder, ph, inst))
                        pattern_insts_without_string_placeholders.add(instantiated_formula)

            logger.debug("Found %d instantiations of pattern after instantiating string placeholders",
                         len(pattern_insts_without_string_placeholders))

            assert all(not get_placeholders(candidate)
                       for candidate in pattern_insts_without_string_placeholders)

            result.update(pattern_insts_without_string_placeholders)

        return result

    def _filter_partial_instantiations(
            self,
            formulas: Iterable[language.Formula],
            inputs_subtrees: Iterable[Dict[Path, language.DerivationTree]]) -> Set[language.Formula]:
        return {
            pattern for pattern in formulas
            if any(not approximately_evaluate_abst_for(
                pattern,
                self.grammar,
                {language.Constant("start", "<start>"): ((), subtrees[()])},
                subtrees).is_false() for subtrees in inputs_subtrees)
        }

    def _apply_filters(
            self,
            formulas: Set[language.Formula],
            filters: List['PatternInstantiationFilter'],
            order: int,
            inputs: List[Dict[Path, language.DerivationTree]],
            parallel: bool = False) -> Set[language.Formula]:
        # TODO: Check when parallel evaluation makes sense. In some cases, like
        #       "test_learn_from_islearn_patterns_file," the overhead of parallel
        #       evaluation seems to bee too high, in other cases (which?), it seems to work.
        for pattern_filter in filters:
            if not pattern_filter.order() == order:
                continue

            if parallel:
                formulas = list(formulas)

                with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                    eval_results = list(pool.map(
                        lambda pattern_inst: int(pattern_filter.predicate(pattern_inst, inputs)),
                        formulas,
                    ))

                formulas = set(itertools.compress(formulas, eval_results))
            else:
                formulas = {
                    pattern_inst for pattern_inst in formulas
                    if pattern_filter.predicate(pattern_inst, inputs)
                }

            logger.debug("%d instantiations remaining after filter '%s'",
                         len(formulas),
                         pattern_filter.name)

        return formulas

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

        logger.info(
            "Reduced positive / negative samples to subsets of %d / %d samples based on k-path coverage.",
            len(self.positive_examples),
            len(self.negative_examples),
        )

        logger.debug(
            "Positive examples:\n%s",
            "\n".join(map(str, self.positive_examples)))

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

    def _instantiate_nonterminal_placeholders(
            self,
            pattern: language.Formula,
            input_reachability_relation: Set[Tuple[str, str]],
            _instantiations: Optional[List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]]] = None
    ) -> Set[language.Formula]:
        if _instantiations:
            instantiations = _instantiations
        else:
            instantiations = self._instantiations_for_placeholder_variables(pattern, input_reachability_relation)

        result: Set[language.Formula] = {
            pattern.substitute_variables(instantiation)
            for instantiation in instantiations
        }

        return result

    def _instantiations_for_placeholder_variables(
            self,
            pattern: language.Formula,
            input_reachability_relation: Set[Tuple[str, str]]
    ) -> List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]]:
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
                previous_instantiations = instantiations
                instantiations = []
                for previous_instantiation in previous_instantiations:
                    for new_instantiation in new_instantiations:
                        # NOTE: There might be clashes, since multiple chains are generated if there is
                        #       more than one variable in a match expression. We have to first check if
                        #       two instantiations happen to conform to each other.
                        if any(new_instantiation[key_1].n_type != previous_instantiation[key_2].n_type
                               for key_1 in new_instantiation
                               for key_2 in previous_instantiation
                               if key_1 == key_2):
                            continue

                        instantiations.append(previous_instantiation | new_instantiation)

        return instantiations

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
                nonterminal_types: Tuple[str, ...] = tuple([var.n_type for var in mexpr_ph.variables])

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
            nonterminal_types: Tuple[str, ...]) -> Set[str]:
        assert all(self.graph.reachable(in_nonterminal, target) for target in nonterminal_types)

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

    def __instantiate_special_predicate_string_placeholders(
            self,
            inst_patterns: Set[language.Formula],
            inputs_subtrees: List[Dict[Path, language.DerivationTree]]) -> Set[language.Formula]:
        result: Set[language.Formula] = set([])

        for pattern in inst_patterns:
            nth_predicates = language.FilterVisitor(
                lambda f: (isinstance(f, language.StructuralPredicateFormula) and
                           f.predicate == isla_predicates.NTH_PREDICATE and
                           isinstance(f.args[0], StringPlaceholderVariable))
            ).collect(pattern)

            if not nth_predicates:
                result.add(pattern)

            # We expect that we don't count in "trivial" nonterminals, i.e., those that immediately
            # unfold to individual characters.
            if any(self._reachable_characters(nth_predicate.args[1].n_type) for nth_predicate in nth_predicates):
                continue

            # For each `nth(<STRING>, elem, container)` add as instantiations all indices
            # of occurrences of `elem`'s nonterminal withint `container`
            sub_result: List[language.Formula] = [pattern]
            nth_predicate: language.StructuralPredicateFormula
            for nth_predicate in nth_predicates:
                elem_nonterminal = nth_predicate.args[1].n_type
                container_nonterminal = nth_predicate.args[2].n_type
                indices: Set[int] = set([])
                for input_subtrees in inputs_subtrees:
                    container_trees = [
                        (path, subtree) for path, subtree in input_subtrees.items()
                        if subtree.value == container_nonterminal]

                    for container_path, container_tree in container_trees:
                        num_occs = len([
                            subtree for path, subtree in input_subtrees.items()
                            if (len(path) > len(container_path) and
                                path[:len(container_path)] == container_path and
                                subtree.value == elem_nonterminal)])

                        indices.update(list(range(1, num_occs + 1)))

                for partial_result in list(sub_result):
                    sub_result.remove(partial_result)

                    def replace_placeholder_by_idx(
                            index: int, subformula: language.Formula) -> language.Formula | bool:
                        if (not isinstance(subformula, language.StructuralPredicateFormula) or
                                subformula.predicate != isla_predicates.NTH_PREDICATE or
                                not isinstance(subformula.args[0], StringPlaceholderVariable)):
                            return False

                        return language.StructuralPredicateFormula(
                            subformula.predicate, str(index), subformula.args[1], subformula.args[2]
                        )

                    sub_result.extend([
                        language.replace_formula(pattern, functools.partial(replace_placeholder_by_idx, index))
                        for index in indices])

            result.update(sub_result)

        return result

    @lru_cache(maxsize=100)
    def _reachable_characters(self, symbol: str) -> Optional[Set[str]]:
        if not is_nonterminal(symbol):
            return {symbol}

        if any(len(expansion) > 1 for expansion in self.canonical_grammar[symbol]):
            return None

        expansion_elements = {
            element for expansion in self.canonical_grammar[symbol]
            for element in expansion
        }

        if all(len(element) == 1 and not is_nonterminal(element)
               for element in expansion_elements):
            return expansion_elements

        children_results = [
            self._reachable_characters(element)
            for element in expansion_elements
        ]

        if any(child_result is None for child_result in children_results):
            return None

        return set(functools.reduce(set.__or__, children_results))

    def _instantiate_string_placeholders(
            self,
            inst_patterns: Set[language.Formula],
            inputs_subtrees: List[Dict[Path, language.DerivationTree]]) -> Set[language.Formula]:

        if all(not isinstance(placeholder, StringPlaceholderVariable)
               for inst_pattern in inst_patterns
               for placeholder in get_placeholders(inst_pattern)):
            return inst_patterns

        # NOTE: To reduce the search space, we also exclude fragments for nonterminals which
        #       have more than `trivial_fragments_exclusion_threshold` terminal children.
        #       This is rather arbitrary, but avoids finding all kinds of spurious invariants
        #       like "there is an 'e' in the text" which are likely to hold any many cases.
        #       10 is the length of a typical <DIGIT> nonterminal expansion set, which
        #       is why those nonterminal fragments would not be pruned.
        trivial_fragments_exclusion_threshold = 10

        many_trivial_terminal_parents: Set[str] = set([])
        for nonterminal in self.grammar:
            reachable = self._reachable_characters(nonterminal)
            if reachable is not None and len(reachable) > trivial_fragments_exclusion_threshold:
                many_trivial_terminal_parents.add(nonterminal)

        fragments: Dict[str, Set[str]] = {nonterminal: set([]) for nonterminal in self.grammar}
        for inp in inputs_subtrees:
            remaining_paths: List[Tuple[Path, language.DerivationTree]] = list(
                sorted(cast(List[Tuple[Path, language.DerivationTree]], list(inp.items())),
                       key=lambda p: len(p[0])))
            handled_paths: Dict[str, Set[Path]] = {nonterminal: set([]) for nonterminal in self.grammar}
            while remaining_paths:
                path, tree = remaining_paths.pop(0)
                if not is_nonterminal(tree.value):
                    continue

                single_child_ancestors: List[language.DerivationTree] = [tree]
                while len(single_child_ancestors[-1].children) == 1:
                    single_child_ancestors.append(single_child_ancestors[-1].children[0])

                if any(child.value in many_trivial_terminal_parents for child in single_child_ancestors):
                    continue

                tree_string = str(tree)
                if not tree_string:
                    continue

                # NOTE: We exclude substrings from fragments; e.g., if we have a <digits>
                #       "1234", don't include the <digits> "34". This might lead
                #       to imprecision, but otherwise the search room tends to explode.
                if any(len(opath) < len(path) and opath == path[:len(opath)]
                       for opath in handled_paths[tree.value]):
                    continue

                handled_paths[tree.value].add(path)
                fragments[tree.value].add(tree_string)
                fragments[tree.value].add(str(len(tree_string)))

                # For strings representing floats, we also include the rounded Integers.
                if is_float(tree_string) and not is_int(tree_string):
                    fragments[tree.value].add(str(int(float(tree_string))))
                    fragments[tree.value].add(str(int(float(tree_string)) + 1))

        logger.debug(
            "Extracted %d language fragments from sample inputs",
            sum(len(value) for value in fragments.values()))

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

                insts: Set[str] = set(functools.reduce(
                    set.__or__, [fragments.get(var.n_type, set([])) for var in non_ph_vars]))

                ph_vars = {v for v in subformula.free_variables() if isinstance(v, StringPlaceholderVariable)}
                sub_result: Set[language.Formula] = {subformula}
                for ph in ph_vars:
                    sub_result = {
                        language.replace_formula(f, functools.partial(substitute_string_placeholder, ph, inst))
                        for f in set(sub_result)
                        for inst in insts
                    }

                return sub_result

            result.update(replace_formula_by_formulas(inst_pattern, replace_placeholder_by_string))

        return result

    def _get_string_placeholder_instantiations(
            self,
            inst_patterns: Set[language.Formula],
            inputs_subtrees: List[Dict[Path, language.DerivationTree]]) -> \
            Dict[language.Formula, Dict[StringPlaceholderVariable, Set[str]]]:
        if all(not isinstance(placeholder, StringPlaceholderVariable)
               for inst_pattern in inst_patterns
               for placeholder in get_placeholders(inst_pattern)):
            return dict.fromkeys(inst_patterns)

        # NOTE: To reduce the search space, we also exclude fragments for nonterminals which
        #       have more than `trivial_fragments_exclusion_threshold` terminal children.
        #       This is rather arbitrary, but avoids finding all kinds of spurious invariants
        #       like "there is an 'e' in the text" which are likely to hold any many cases.
        #       10 is the length of a typical <DIGIT> nonterminal expansion set, which
        #       is why those nonterminal fragments would not be pruned.
        trivial_fragments_exclusion_threshold = 10

        many_trivial_terminal_parents: Set[str] = set([])
        for nonterminal in self.grammar:
            reachable = self._reachable_characters(nonterminal)
            if reachable is not None and len(reachable) > trivial_fragments_exclusion_threshold:
                many_trivial_terminal_parents.add(nonterminal)

        fragments: Dict[str, Set[str]] = {nonterminal: set([]) for nonterminal in self.grammar}
        for inp in inputs_subtrees:
            remaining_paths: List[Tuple[Path, language.DerivationTree]] = list(
                sorted(cast(List[Tuple[Path, language.DerivationTree]], list(inp.items())),
                       key=lambda p: len(p[0])))
            handled_paths: Dict[str, Set[Path]] = {nonterminal: set([]) for nonterminal in self.grammar}
            while remaining_paths:
                path, tree = remaining_paths.pop(0)
                if not is_nonterminal(tree.value):
                    continue

                single_child_ancestors: List[language.DerivationTree] = [tree]
                while len(single_child_ancestors[-1].children) == 1:
                    single_child_ancestors.append(single_child_ancestors[-1].children[0])

                if any(child.value in many_trivial_terminal_parents for child in single_child_ancestors):
                    continue

                tree_string = str(tree)
                if not tree_string:
                    continue

                # NOTE: We exclude substrings from fragments; e.g., if we have a <digits>
                #       "1234", don't include the <digits> "34". This might lead
                #       to imprecision, but otherwise the search room tends to explode.
                if any(len(opath) < len(path) and opath == path[:len(opath)]
                       for opath in handled_paths[tree.value]):
                    continue

                handled_paths[tree.value].add(path)
                fragments[tree.value].add(tree_string)
                fragments[tree.value].add(str(len(tree_string)))

                # For strings representing floats, we also include the rounded Integers.
                if is_float(tree_string) and not is_int(tree_string):
                    fragments[tree.value].add(str(int(float(tree_string))))
                    fragments[tree.value].add(str(int(float(tree_string)) + 1))

        logger.debug(
            "Extracted %d language fragments from sample inputs",
            sum(len(value) for value in fragments.values()))

        result: Dict[language.Formula, Dict[StringPlaceholderVariable, Set[str]]] = {}
        for inst_pattern in inst_patterns:
            def extract_instantiations(subformula: language.Formula) -> None:
                nonlocal result

                if not any(isinstance(arg, StringPlaceholderVariable) for arg in subformula.free_variables()):
                    return

                non_ph_vars = {v for v in subformula.free_variables() if not isinstance(v, PlaceholderVariable)}

                insts: Set[str] = set(functools.reduce(
                    set.__or__, [fragments.get(var.n_type, set([])) for var in non_ph_vars]))

                ph_vars = {v for v in subformula.free_variables() if isinstance(v, StringPlaceholderVariable)}
                for ph in ph_vars:
                    result.setdefault(inst_pattern, {}).setdefault(ph, set([])).update(insts)

            class InstantiationVisitor(language.FormulaVisitor):
                def visit_smt_formula(self, formula: language.SMTFormula):
                    extract_instantiations(formula)

                def visit_predicate_formula(self, formula: language.StructuralPredicateFormula):
                    extract_instantiations(formula)

                def visit_semantic_predicate_formula(self, formula: language.SemanticPredicateFormula):
                    extract_instantiations(formula)

            inst_pattern.accept(InstantiationVisitor())

        return result


def substitute_string_placeholder(
        ph: PlaceholderVariable,
        inst: str,
        formula: language.Formula) -> language.Formula | bool:
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


def approximately_evaluate_abst_for(
        formula: language.Formula,
        grammar: Grammar,
        assignments: Dict[language.Variable, Tuple[Path, language.DerivationTree] | str],
        subtrees: Dict[Path, language.DerivationTree]) -> ThreeValuedTruth:
    # TODO: Handle String placeholder variables in predicate formulas
    if isinstance(formula, language.SMTFormula):
        if any(isinstance(arg, PlaceholderVariable) and arg not in assignments
               for arg in formula.free_variables()):
            return ThreeValuedTruth.unknown()

        if any(var not in assignments for var in formula.free_variables()):
            # This happens for quantified formulas over integers: We simply
            # strip of the quantifier, and a variable remains.
            return ThreeValuedTruth.unknown()

        try:
            translation = evaluate_z3_expression(formula.formula)
            var_map: Dict[str, language.Variable] = {
                var.name: var
                for var in assignments
            }

            # args_instantiation = tuple([
            #     str(assignments[var_map[arg]][1])
            #     for arg in translation[0]])

            args_instantiation = ()
            for arg in translation[0]:
                var = var_map[arg]
                if isinstance(var, StringPlaceholderVariable):
                    args_instantiation += (assignments[var],)
                else:
                    args_instantiation += (str(assignments[var][1]),)

            return ThreeValuedTruth.from_bool(
                translation[1](args_instantiation) if args_instantiation
                else translation[1])
        except DomainError:
            return ThreeValuedTruth.false()
        except NotImplementedError:
            return is_valid(z3.substitute(
                formula.formula,
                *tuple({z3.String(symbol.name): z3.StringVal(str(symbol_assignment[1]))
                        for symbol, symbol_assignment
                        in assignments.items()}.items())))

    elif isinstance(formula, language.NumericQuantifiedFormula):
        return approximately_evaluate_abst_for(
            formula.inner_formula, grammar, assignments, subtrees)
    elif isinstance(formula, language.QuantifiedFormula):
        assert isinstance(formula.in_variable, language.Variable)
        assert formula.in_variable in assignments
        in_path, in_inst = assignments[formula.in_variable]

        if formula.bind_expression is None:
            paths = list(subtrees.keys())
            in_path_idx = paths.index(in_path)
            new_assignments: List[Dict[language.Variable, Tuple[Path, language.DerivationTree]]] = []
            root_path_indices: List[int] = []
            for path_idx, path in enumerate(paths[in_path_idx + 1:]):
                if len(path) <= len(in_path):
                    break

                if subtrees[path].value == formula.bound_variable.n_type:
                    new_assignments.append({formula.bound_variable: (path, subtrees[path])})
                    root_path_indices.append(path_idx + in_path_idx + 1)
        else:
            paths = list(subtrees.keys())
            in_path_idx = paths.index(in_path)
            sub_paths = {in_path: in_inst}
            for path in paths[in_path_idx + 1:]:
                if len(path) <= len(in_path):
                    break

                sub_paths[path] = subtrees[path]

            new_assignments = matches_for_quantified_formula(formula, grammar, subtrees[()], {}, paths=sub_paths)

        new_assignments = [
            new_assignment | assignments
            for new_assignment in new_assignments]

        if not new_assignments:
            return ThreeValuedTruth.false()

        if isinstance(formula, language.ExistsFormula):
            return ThreeValuedTruth.from_bool(any(
                not approximately_evaluate_abst_for(formula.inner_formula, grammar, new_assignment, subtrees).is_false()
                for new_assignment in new_assignments))
        else:
            return ThreeValuedTruth.from_bool(all(
                not approximately_evaluate_abst_for(formula.inner_formula, grammar, new_assignment, subtrees).is_false()
                for new_assignment in new_assignments))
    elif isinstance(formula, language.StructuralPredicateFormula):
        if any(isinstance(arg, PlaceholderVariable) for arg in formula.args):
            return ThreeValuedTruth.unknown()

        arg_insts = [
            arg if isinstance(arg, str)
            else next(path for path, subtree in subtrees.items() if subtree.id == arg.id)
            if isinstance(arg, language.DerivationTree)
            else assignments[arg][0]
            for arg in formula.args]

        return ThreeValuedTruth.from_bool(formula.predicate.evaluate(subtrees[()], *arg_insts))
    elif isinstance(formula, language.SemanticPredicateFormula):
        if any(isinstance(arg, PlaceholderVariable) for arg in formula.args):
            return ThreeValuedTruth.unknown()

        arg_insts = [arg if isinstance(arg, language.DerivationTree) or arg not in assignments
                     else assignments[arg][1]
                     for arg in formula.args]
        eval_res = formula.predicate.evaluate(grammar, *arg_insts)

        if eval_res.true():
            return ThreeValuedTruth.true()
        elif eval_res.false():
            return ThreeValuedTruth.false()

        if not eval_res.ready() or not all(isinstance(key, language.Constant) for key in eval_res.result):
            # Evaluation resulted in a tree update; that is, the formula is satisfiable, but only
            # after an update of its arguments. This result happens when evaluating formulas during
            # solution search after instantiating variables with concrete trees.
            return ThreeValuedTruth.unknown()

        assignments.update({const: (tuple(), assgn) for const, assgn in eval_res.result.items()})
        return ThreeValuedTruth.true()
    elif isinstance(formula, language.NegatedFormula):
        return ThreeValuedTruth.not_(approximately_evaluate_abst_for(
            formula.args[0], grammar, assignments, subtrees))
    elif isinstance(formula, language.ConjunctiveFormula):
        # Relaxation: Unknown is OK, only False is excluded.
        return ThreeValuedTruth.from_bool(all(
            not approximately_evaluate_abst_for(sub_formula, grammar, assignments, subtrees).is_false()
            for sub_formula in formula.args))
    elif isinstance(formula, language.DisjunctiveFormula):
        return ThreeValuedTruth.from_bool(any(
            not approximately_evaluate_abst_for(sub_formula, grammar, assignments, subtrees).is_false()
            for sub_formula in formula.args))
    else:
        raise NotImplementedError()


class PatternInstantiationFilter(ABC):
    def __init__(self, name: str):
        self.name = name

    def order(self) -> int:
        """Currently: If 0, can be executed after all nonterminal types in quantifiers are fixed.
        If 1, can only be executed after also all nonterminal string placeholders (in predicates)
        have been instantiated. If 2, can only be executed after also all string placeholders
        have been instantiated."""
        raise NotImplementedError()

    def __eq__(self, other):
        return isinstance(other, PatternInstantiationFilter) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def predicate(self, formula: language.Formula, inputs: List[Dict[Path, language.DerivationTree]]) -> bool:
        raise NotImplementedError()


class NonterminalStringInCountPredicatesFilter(PatternInstantiationFilter):
    def __init__(self, graph: gg.GrammarGraph, input_reachability_relation: Set[Tuple[str, str]]):
        super().__init__("Nonterminal String in `count` Predicates Filter")
        self.graph = graph
        self.input_reachability_relation = input_reachability_relation

    def order(self) -> int:
        return 1

    def reachable(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return reachable(self.graph, from_nonterminal, to_nonterminal)

    def reachable_in_inputs(self, from_nonterminal: str, to_nonterminal: str) -> bool:
        return (from_nonterminal, to_nonterminal) in self.input_reachability_relation

    def predicate(self, formula: language.Formula, _: List[Dict[Path, language.DerivationTree]]) -> bool:
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


class InVisitor(language.FormulaVisitor):
    def __init__(self):
        self.result: Set[Tuple[language.Variable, language.Variable]] = set()

    def visit_exists_formula(self, formula: language.ExistsFormula):
        self.handle(formula)

    def visit_forall_formula(self, formula: language.ForallFormula):
        self.handle(formula)

    def handle(self, formula: language.QuantifiedFormula):
        self.result.add((formula.bound_variable, formula.in_variable))
        if formula.bind_expression:
            if any(isinstance(elem, MexprPlaceholderVariable)
                   for elem in formula.bind_expression.bound_elements):
                phs = [elem for elem in formula.bind_expression.bound_elements
                       if isinstance(elem, MexprPlaceholderVariable)]
                assert len(phs) == 1
                ph = cast(MexprPlaceholderVariable, phs[0])
                for nonterminal_placeholder in ph.variables:
                    self.result.add((nonterminal_placeholder, formula.bound_variable))
            else:
                self.result.update({
                    (var, formula.bound_variable)
                    for var in formula.bind_expression.bound_elements
                    if isinstance(var, language.BoundVariable)
                       and not isinstance(var, language.DummyVariable)
                })


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
    def __init__(self, data: Dict[str, List[Dict[str, str]]]):
        self.groups: Dict[str, Dict[str, language.Formula]] = {}
        for group_name, elements in data.items():
            for entry in elements:
                name = entry["name"]
                constraint = parse_abstract_isla(entry["constraint"])
                self.groups.setdefault(group_name, {})[name] = constraint

    def __getitem__(self, item: str) -> Set[language.Formula]:
        if item in self.groups:
            return set(self.groups[item].values())

        for elements in self.groups.values():
            if item in elements:
                return {elements[item]}

        logger.warning(f"Could not find pattern for query {item}.")
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
                result += f"[[{group}]]\n\n"
                result += f'name = "{name}"\n'
                result += "constraint = '''\n"
                result += AbstractISLaUnparser(self.groups[group][name]).unparse()
                result += "\n'''"
                result += "\n\n"

        return result.strip()


def create_input_reachability_relation(inputs: Iterable[language.DerivationTree]) -> Set[Tuple[str, str]]:
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


def patterns_from_file(file_name: str = STANDARD_PATTERNS_REPO) -> PatternRepository:
    if os.path.isfile(file_name):
        f = open(file_name, "r")
        contents = f.read()
        f.close()
    else:
        contents = pkgutil.get_data("islearn", STANDARD_PATTERNS_REPO).decode("UTF-8")

    data: Dict[str, List[Dict[str, str]]] = cast(Dict[str, List[Dict[str, str]]], toml.loads(contents))
    assert isinstance(data, dict)
    assert len(data) > 0
    assert all(isinstance(entry, str) for entry in data.keys())
    assert all(isinstance(entry, list) for entry in data.values())

    return PatternRepository(data)
