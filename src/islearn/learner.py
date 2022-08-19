import copy
import functools
import inspect
import itertools
import logging
import os.path
import pkgutil
import string
from abc import ABC
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional, cast, Callable, Iterable, Sequence

import datrie
import isla.fuzzer
import toml
import z3
from grammar_graph import gg
from isla import language, isla_predicates
from isla.evaluator import evaluate, matches_for_quantified_formula
from isla.helpers import RE_NONTERMINAL, weighted_geometric_mean, \
    is_nonterminal, dict_of_lists_to_list_of_dicts, canonical
from isla.isla_predicates import reachable
from isla.language import set_smt_auto_eval, ensure_unique_bound_variables
from isla.solver import ISLaSolver
from isla.three_valued_truth import ThreeValuedTruth
from isla.trie import path_to_trie_key
from isla.type_defs import Grammar, ParseTree, Path
from isla.z3_helpers import z3_subst, evaluate_z3_expression, is_valid, \
    DomainError
from pathos import multiprocessing as pmp

from islearn.helpers import connected_chains, transitive_closure, tree_in, \
    is_int, is_float, e_assert
from islearn.language import NonterminalPlaceholderVariable, PlaceholderVariable, \
    NonterminalStringPlaceholderVariable, parse_abstract_isla, StringPlaceholderVariable, \
    AbstractISLaUnparser, MexprPlaceholderVariable, AbstractBindExpression, DisjunctiveStringsPlaceholderVariable, \
    StringPlaceholderVariableTypes
from islearn.mutation import MutationFuzzer
from islearn.parse_tree_utils import replace_path, expand_tree, tree_leaves, \
    get_subtree, tree_paths, trie_from_parse_tree, next_trie_key, tree_from_paths, Tree, get_subtrie
from islearn.reducer import InputReducer

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
            graph: gg.GrammarGraph,
            columns_parallel: bool = False,
            lazy: bool = False,
            result_threshold: float = .9) -> 'TruthTableRow':
        """If lazy is True, then the evaluation stops as soon as result_threshold can no longer be
        reached. E.g., if result_threshold is .9 and there are 100 inputs, then after more than
        10 negative results, 90% positive results is no longer possible."""

        if columns_parallel:
            with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                iterator = pool.imap(
                    lambda inp: evaluate(self.formula, inp, graph.grammar, graph=graph).is_true(),
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

                eval_result = evaluate(self.formula, inp, graph.grammar, graph=graph).is_true()
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

    def __len__(self):
        return len(self.eval_results)

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
        self.__row_hashes = set([])
        self.__rows = []
        for row in rows:
            row_hash = hash(row)
            if row_hash not in self.__row_hashes:
                self.__row_hashes.add(row_hash)
                self.__rows.append(row)

    def __repr__(self):
        return f"TruthTable({repr(self.__rows)})"

    def __str__(self):
        return "\n".join(map(str, self.__rows))

    def __getitem__(self, item: int | language.Formula) -> TruthTableRow:
        if isinstance(item, int):
            return self.__rows[item]

        assert isinstance(item, language.Formula)

        try:
            return next(row for row in self.__rows if row.formula == item)
        except StopIteration:
            raise KeyError(item)

    def __len__(self):
        return len(self.__rows)

    def __iter__(self):
        return iter(self.__rows)

    def append(self, row: TruthTableRow):
        row_hash = hash(row)
        if row_hash not in self.__row_hashes:
            self.__row_hashes.add(row_hash)
            self.__rows.append(row)

    def remove(self, row: TruthTableRow):
        self.__rows.remove(row)
        self.__row_hashes.remove(hash(row))

    def __add__(self, other: 'TruthTable') -> 'TruthTable':
        return TruthTable(self.__rows + other.__rows)

    def __iadd__(self, other: 'TruthTable') -> 'TruthTable':
        for row in other.__rows:
            self.append(row)

        return self

    def evaluate(
            self,
            graph: gg.GrammarGraph,
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
                self.__rows = set(pool.map(
                    lambda row: row.evaluate(graph, columns_parallel, lazy=lazy, result_threshold=result_threshold),
                    self.__rows
                ))
        else:
            for row in self.__rows:
                row.evaluate(graph, columns_parallel, lazy=lazy, result_threshold=result_threshold)

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
            mexpr_expansion_limit: int = 1,
            max_nonterminals_in_mexpr: Optional[int] = None,
            min_recall: float = .9,
            min_specificity: float = .6,
            max_disjunction_size: int = 1,
            max_conjunction_size: int = 2,
            exclude_nonterminals: Optional[Iterable[str]] = None,
            include_negations_in_disjunctions: bool = False,
            reduce_inputs_for_learning: bool = True,
            reduce_all_inputs: bool = False,
            generate_new_learning_samples: bool = True,
            do_generate_more_inputs: bool = True,
            filter_inputs_for_learning_by_kpaths: bool = True,
    ):
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
        self.max_nonterminals_in_mexpr = max_nonterminals_in_mexpr
        self.min_recall = min_recall
        self.min_specificity = min_specificity
        self.max_disjunction_size = max_disjunction_size
        self.max_conjunction_size = max_conjunction_size
        self.exclude_nonterminals = exclude_nonterminals or set([])
        self.include_negations_in_disjunctions = include_negations_in_disjunctions
        self.reduce_inputs_for_learning = reduce_inputs_for_learning
        self.reduce_all_inputs = reduce_all_inputs
        self.generate_new_learning_samples = generate_new_learning_samples
        self.do_generate_more_inputs = do_generate_more_inputs
        self.filter_inputs_for_learning_by_kpaths = filter_inputs_for_learning_by_kpaths

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

    def learn_invariants(self, ensure_unique_var_names: bool = True) -> Dict[language.Formula, Tuple[float, float]]:
        if self.prop and self.do_generate_more_inputs:
            self._generate_more_inputs()
            assert len(self.positive_examples) > 0, "Cannot learn without any positive examples!"
            assert all(self.prop(positive_example) for positive_example in self.positive_examples)
            assert all(not self.prop(negative_example) for negative_example in self.negative_examples)

        if self.reduce_all_inputs and self.prop is not None:
            logger.info(
                "Reducing %d positive samples w.r.t. property and k=%d.",
                len(self.positive_examples),
                self.k)
            reducer = InputReducer(self.grammar, self.prop, self.k)
            self.positive_examples = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.positive_examples]

            logger.info(
                "Reducing %d negative samples w.r.t. property and k=%d.",
                len(self.negative_examples),
                self.k)
            reducer = InputReducer(self.grammar, lambda t: not self.prop(t), self.k)
            self.negative_examples = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.negative_examples]

        if self.generate_new_learning_samples or not self.original_positive_examples:
            self.positive_examples_for_learning = \
                self._sort_inputs(
                    self.positive_examples,
                    self.filter_inputs_for_learning_by_kpaths,
                    more_paths_weight=1.7,
                    smaller_inputs_weight=1.0)[:self.target_number_positive_samples_for_learning]
        else:
            self.positive_examples_for_learning = \
                self._sort_inputs(
                    self.original_positive_examples,
                    self.filter_inputs_for_learning_by_kpaths,
                    more_paths_weight=1.7,
                    smaller_inputs_weight=1.0)[:self.target_number_positive_samples_for_learning]

        logger.info(
            "Keeping %d positive examples for candidate generation.",
            len(self.positive_examples_for_learning)
        )

        if ((not self.reduce_all_inputs or not self.generate_new_learning_samples)
                and self.reduce_inputs_for_learning
                and self.prop is not None):
            logger.info(
                "Reducing %d inputs for learning w.r.t. property and k=%d.",
                len(self.positive_examples_for_learning),
                self.k
            )
            reducer = InputReducer(self.grammar, self.prop, self.k)
            self.positive_examples_for_learning = [
                reducer.reduce_by_smallest_subtree_replacement(inp)
                for inp in self.positive_examples_for_learning]

        logger.debug(
            "Examples for learning:\n%s",
            "\n".join(map(str, self.positive_examples_for_learning)))

        candidates = self.generate_candidates(self.patterns, self.positive_examples_for_learning)
        logger.info("Found %d invariant candidates.", len(candidates))

        logger.debug(
            "Candidates:\n%s",
            "\n\n".join([language.ISLaUnparser(candidate).unparse() for candidate in candidates]))

        logger.info("Filtering invariants.")

        # Only consider *real* invariants

        # NOTE: Disabled parallel evaluation for now. In certain cases, this renders
        #       the filtering process *much* slower, or gives rise to stack overflows
        #       (e.g., "test_learn_from_islearn_patterns_file" example).
        recall_truth_table = TruthTable([
            TruthTableRow(inv, self.positive_examples)
            for inv in candidates
        ]).evaluate(
            self.graph,
            # rows_parallel=True,
            lazy=self.max_disjunction_size < 2,
            result_threshold=self.min_recall
        )

        if self.max_disjunction_size < 2:
            for row in recall_truth_table:
                if row.eval_result() < self.min_recall:
                    recall_truth_table.remove(row)

        precision_truth_table = None
        if self.negative_examples:
            logger.info("Evaluating precision.")
            logger.debug("Negative samples:\n" + "\n-----------\n".join(map(str, self.negative_examples)))

            precision_truth_table = TruthTable([
                TruthTableRow(row.formula, self.negative_examples)
                for row in recall_truth_table
            ]).evaluate(
                self.graph,
                # rows_parallel=True
            )

            assert len(precision_truth_table) == len(recall_truth_table)

        assert not self.negative_examples or precision_truth_table is not None

        invariants = {
            row.formula for row in recall_truth_table
            if row.eval_result() >= self.min_recall
        }

        logger.info(
            "%d invariants with recall >= %d%% remain after filtering.",
            len(invariants),
            int(self.min_recall * 100))

        logger.debug("Invariants:\n%s", "\n\n".join(map(lambda f: language.ISLaUnparser(f).unparse(), invariants)))

        if self.max_disjunction_size > 1:
            logger.info("Calculating recall of Boolean combinations.")

            disjunctive_recall_truthtable = copy.deepcopy(recall_truth_table)
            assert precision_truth_table is None or len(disjunctive_recall_truthtable) == len(precision_truth_table)

            for level in range(2, self.max_disjunction_size + 1):
                assert precision_truth_table is None or len(disjunctive_recall_truthtable) == len(precision_truth_table)
                logger.debug(f"Disjunction size: {level}")

                for rows_with_indices in itertools.combinations(enumerate(recall_truth_table), level):
                    assert (precision_truth_table is None or
                            len(disjunctive_recall_truthtable) == len(precision_truth_table))
                    assert (precision_truth_table is None or
                            all(rwi[1].formula == precision_truth_table[rwi[0]].formula for rwi in rows_with_indices))

                    max_num_negations = level // 2 if self.include_negations_in_disjunctions else 0
                    for formulas_to_negate in (
                            t for t in itertools.product(*[[0, 1] for _ in range(3)]) if sum(t) <= max_num_negations):

                        # To ensure that only "meaningful" properties are negated, the un-negated properties
                        # should hold for at least 20% of all inputs; but at most for 80%, since otherwise,
                        # the negation is overly specific.
                        negated_rows: List[Tuple[int, TruthTableRow]] = list(
                            itertools.compress(rows_with_indices, formulas_to_negate))
                        if any(.8 < negated_row.eval_result() < .2 for _, negated_row in negated_rows):
                            continue

                        recall_table_rows = [
                            -row if bool(negate) else row
                            for negate, (_, row) in zip(formulas_to_negate, rows_with_indices)]

                        # Compute recall of disjunction, add if above threshold and
                        # an improvement over all participants of the disjunction
                        disjunction = functools.reduce(TruthTableRow.__or__, recall_table_rows)
                        new_recall = disjunction.eval_result()
                        if (new_recall < self.min_recall or
                                not all(new_recall > row.eval_result() for row in recall_table_rows)):
                            continue

                        disjunctive_recall_truthtable.append(disjunction)

                        if precision_truth_table is not None:
                            # Also add disjunction to the precision truth table. Saves us a couple of evaluations.
                            precision_table_rows = [
                                -precision_truth_table[idx] if bool(negate) else precision_truth_table[idx]
                                for negate, (idx, _) in zip(formulas_to_negate, rows_with_indices)]
                            disjunction = functools.reduce(TruthTableRow.__or__, precision_table_rows)
                            precision_truth_table.append(disjunction)

            recall_truth_table = disjunctive_recall_truthtable

            invariants = {
                row.formula for row in recall_truth_table
                if row.eval_result() >= self.min_recall
            }

            logger.info(
                "%d invariants with recall >= %d%% remain after building Boolean combinations.",
                len(invariants),
                int(self.min_recall * 100))

            # logger.debug(
            #   "Invariants:\n%s", "\n\n".join(map(lambda f: language.ISLaUnparser(f).unparse(), invariants)))

        # assert all(evaluate(row.formula, inp, self.grammar).is_true()
        #            for inp in self.positive_examples
        #            for row in recall_truth_table
        #            if row.eval_result() == 1)

        if not self.negative_examples:
            # TODO: Enforce unique names, sort
            if ensure_unique_var_names:
                invariants = sorted(list(map(ensure_unique_bound_variables, invariants)), key=lambda inv: len(inv))
            else:
                invariants = sorted(list(invariants), key=lambda inv: len(inv))
            return {
                row.formula: (1.0, row.eval_result())
                for row in recall_truth_table
                if row.eval_result() >= self.min_recall
            }

        indices_to_remove = list(reversed([
            idx for idx, row in enumerate(recall_truth_table)
            if row.eval_result() < self.min_recall]))
        assert sorted(indices_to_remove, reverse=True) == indices_to_remove
        for idx in indices_to_remove:
            recall_truth_table.remove(recall_truth_table[idx])
            precision_truth_table.remove(precision_truth_table[idx])

        logger.info("Calculating precision of Boolean combinations.")
        conjunctive_precision_truthtable = copy.copy(precision_truth_table)
        for level in range(2, self.max_conjunction_size + 1):
            logger.debug(f"Conjunction size: {level}")
            assert len(recall_truth_table) == len(conjunctive_precision_truthtable)
            for rows_with_indices in itertools.combinations(enumerate(precision_truth_table), level):
                precision_table_rows = [row for (_, row) in rows_with_indices]

                # Only consider combinations where all rows meet minimum recall requirement.
                # Recall doesn't get better by forming conjunctions!
                if any(recall_truth_table[idx].eval_result() < self.min_recall
                       for idx, _ in rows_with_indices):
                    continue

                # Compute precision of conjunction, add if above threshold and
                # an improvement over all participants of the conjunction
                conjunction = functools.reduce(TruthTableRow.__and__, precision_table_rows)
                new_precision = 1 - conjunction.eval_result()
                if (new_precision < self.min_specificity or
                        not all(new_precision > 1 - row.eval_result() for row in precision_table_rows)):
                    continue

                conjunctive_precision_truthtable.append(conjunction)

                recall_table_rows = [recall_truth_table[idx] for idx, _ in rows_with_indices]
                conjunction = functools.reduce(TruthTableRow.__and__, recall_table_rows)
                recall_truth_table.append(conjunction)

        precision_truth_table = conjunctive_precision_truthtable

        # assert all(evaluate(row.formula, inp, self.grammar).is_false()
        #            for inp in self.negative_examples
        #            for row in precision_truth_table
        #            if row.eval_result() == 0)

        result: Dict[language.Formula, Tuple[float, float]] = {
            precision_row.formula if not ensure_unique_var_names
            else language.ensure_unique_bound_variables(precision_row.formula):
                (1 - precision_row.eval_result(), recall_truth_table[idx].eval_result())
            for idx, precision_row in enumerate(precision_truth_table)
            if (1 - precision_row.eval_result() >= self.min_specificity and
                recall_truth_table[idx].eval_result() >= self.min_recall)
        }

        logger.info(
            "Found %d invariants with precision >= %d%%.",
            len([p for p in result.values() if p[0] >= self.min_specificity]),
            int(self.min_specificity * 100),
        )

        # TODO: Sort within same recall / specificity values: Fewer disjunctions,
        #       more common String constants... To optimize specificity further.

        return dict(
            cast(List[Tuple[language.Formula, Tuple[float, float]]],
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

        tries: List[datrie.Trie] = [inp.trie().trie for inp in inputs]

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

            # NOTE: At this point, filtering is not useful. It is cheaper to first instantiate
            #       match expressions (if any), which reduces the search space, and to filter then.
            # pattern_insts_without_nonterminal_placeholders = self._filter_partial_instantiations(
            #     pattern_insts_without_nonterminal_placeholders, tries)
            # logger.debug("%d instantiations remain after filtering",
            #              len(pattern_insts_without_nonterminal_placeholders))

            # 2. Match expression placeholders
            pattern_insts_without_mexpr_placeholders = self._instantiate_mexpr_placeholders(
                pattern_insts_without_nonterminal_placeholders)

            logger.debug("Found %d instantiations of pattern after instantiating match expression placeholders",
                         len(pattern_insts_without_mexpr_placeholders))

            pattern_insts_without_mexpr_placeholders = self._filter_partial_instantiations(
                pattern_insts_without_mexpr_placeholders, tries)
            logger.debug("%d instantiations remain after filtering",
                         len(pattern_insts_without_mexpr_placeholders))

            # 3. Special string placeholders in predicates.
            #    This comprises, e.g., `nth(<STRING>, elem, container`.
            pattern_insts_without_special_string_placeholders = \
                self.__instantiate_special_predicate_string_placeholders(
                    pattern_insts_without_mexpr_placeholders, tries)

            logger.debug("Found %d instantiations of pattern after instantiating special predicate string placeholders",
                         len(pattern_insts_without_special_string_placeholders))

            if pattern_insts_without_special_string_placeholders != pattern_insts_without_mexpr_placeholders:
                pattern_insts_without_special_string_placeholders = self._filter_partial_instantiations(
                    pattern_insts_without_special_string_placeholders, tries)
                logger.debug("%d instantiations remain after filtering",
                             len(pattern_insts_without_special_string_placeholders))

            # 4. Nonterminal-String placeholders
            pattern_insts_without_nonterminal_string_placeholders = self._instantiate_nonterminal_string_placeholders(
                pattern_insts_without_special_string_placeholders)

            pattern_insts_without_nonterminal_string_placeholders = self._apply_filters(
                pattern_insts_without_nonterminal_string_placeholders, filters, 1, tries)

            logger.debug("Found %d instantiations of pattern after instantiating nonterminal string placeholders",
                         len(pattern_insts_without_nonterminal_string_placeholders))

            if (pattern_insts_without_special_string_placeholders !=
                    pattern_insts_without_nonterminal_string_placeholders):
                pattern_insts_without_nonterminal_string_placeholders = self._filter_partial_instantiations(
                    pattern_insts_without_nonterminal_string_placeholders, tries)
                logger.debug("%d instantiations remain after filtering",
                             len(pattern_insts_without_nonterminal_string_placeholders))

            # 5. String placeholders
            if not any(isinstance(ph, StringPlaceholderVariableTypes) for ph in get_placeholders(pattern)):
                pattern_insts_without_string_placeholders = pattern_insts_without_nonterminal_string_placeholders
            else:
                pattern_insts_without_string_placeholders = self._instantiate_string_placeholders(
                    pattern_insts_without_nonterminal_string_placeholders, tries)

                logger.debug("Found %d instantiations of pattern after instantiating string placeholders",
                             len(pattern_insts_without_string_placeholders))

            assert all(not get_placeholders(candidate)
                       for candidate in pattern_insts_without_string_placeholders)

            result.update(pattern_insts_without_string_placeholders)

        return result

    def _filter_partial_instantiations(
            self,
            formulas: Iterable[language.Formula],
            tries: Iterable[datrie.Trie]) -> Set[language.Formula]:
        result: Set[language.Formula] = set()

        for pattern in formulas:
            for trie in tries:
                if not approximately_evaluate_abst_for(
                        pattern,
                        self.grammar,
                        self.graph,
                        {language.Constant("start", "<start>"): trie[path_to_trie_key(())]},
                        trie).is_false():
                    result.add(pattern)
                    break

        return result

    def _apply_filters(
            self,
            formulas: Set[language.Formula],
            filters: List['PatternInstantiationFilter'],
            order: int,
            tries: List[datrie.Trie],
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
                        lambda pattern_inst: int(pattern_filter.predicate(pattern_inst, tries)),
                        formulas,
                    ))

                formulas = set(itertools.compress(formulas, eval_results))
            else:
                formulas = {
                    pattern_inst for pattern_inst in formulas
                    if pattern_filter.predicate(pattern_inst, tries)
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

        assert len(self.positive_examples) > 0, "Cannot learn without any positive examples!"
        pe_before = len(self.positive_examples)
        ne_before = len(self.negative_examples)
        mutation_fuzzer = MutationFuzzer(self.grammar, self.positive_examples, self.prop, k=self.k)
        for inp in mutation_fuzzer.run(
                num_iterations=min(
                    30,
                    max(self.target_number_positive_samples - pe_before, 30),
                    max(self.target_number_negative_samples - ne_before, 30)
                ),
                alpha=.1, yield_negative=True):
            if self.prop(inp) and not tree_in(inp, self.positive_examples):
                self.positive_examples.append(inp)
            elif not self.prop(inp) and not tree_in(inp, self.negative_examples):
                self.negative_examples.append(inp)

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

        self.positive_examples = self._sort_inputs(
            self.positive_examples,
            self.filter_inputs_for_learning_by_kpaths,
            more_paths_weight=1.5,
            smaller_inputs_weight=1.0,
        )[:self.target_number_positive_samples]

        self.negative_examples = \
            self._sort_inputs(
                self.negative_examples,
                self.filter_inputs_for_learning_by_kpaths,
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

    def _sort_inputs(
            self,
            inputs: Iterable[language.DerivationTree],
            filter_inputs_for_learning_by_kpaths: bool,
            more_paths_weight: float = 1.0,
            smaller_inputs_weight: float = 0.0) -> List[language.DerivationTree]:
        assert more_paths_weight or smaller_inputs_weight
        inputs = set(inputs)
        result: List[language.DerivationTree] = []

        tree_paths = {
            inp: {
                path for path in self.graph.k_paths_in_tree(inp.to_parse_tree(), self.k)
                if (not isinstance(path[-1], gg.TerminalNode) or
                    (not isinstance(path[-1], gg.TerminalNode) and len(path[-1].symbol) > 1))
            } for inp in inputs}

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
            inputs.remove(inp)
            uncovered = uncovered_paths(inp)

            if filter_inputs_for_learning_by_kpaths and not uncovered:
                continue

            covered_paths.update(uncovered)
            result.append(inp)

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
        instantiations: List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]] = []

        in_visitor = InVisitor()
        pattern.accept(in_visitor)
        variable_chains: Set[Tuple[language.Variable, ...]] = connected_chains(in_visitor.result)
        start_const = extract_top_level_constant(pattern)
        assert all(chain[-1] == start_const for chain in variable_chains)
        reachable = {fr: [to for fr_, to in input_reachability_relation
                          if fr == fr_ and to not in self.exclude_nonterminals] for fr in self.grammar}

        # Combine variable chains to a tree structure to avoid conflicting instantiations.
        InstantiationTree = Tree[Tuple[language.Variable, Optional[str]]]
        initial_tree: InstantiationTree = tree_from_paths(
            [list(reversed(chain)) for chain in variable_chains])

        # We basically perform a BFS over the partially instantiated trees and
        # instantiate children based on the parent values.
        stack: List[Tuple[
            List[InstantiationTree],
            Dict[NonterminalPlaceholderVariable, language.BoundVariable]]] = \
            [([tree for _, tree in tree_paths(initial_tree)], {})]
        while stack:
            remaining_subtrees, inst_map = stack.pop()
            if not remaining_subtrees:
                instantiations.append(inst_map)
                continue

            parent_variable: PlaceholderVariable
            children: List[InstantiationTree]
            (parent_variable, children), *remaining_subtrees = remaining_subtrees

            if not children:
                stack.append((remaining_subtrees, inst_map))
                continue

            if isinstance(parent_variable, PlaceholderVariable):
                assert parent_variable in inst_map
                parent_instantiation = inst_map[parent_variable].n_type
            else:
                parent_instantiation = parent_variable.n_type

            reachable_nonterminals = reachable[parent_instantiation]
            if not reachable_nonterminals:
                continue

            for instantiation in itertools.product(*[reachable_nonterminals for _ in range(len(children))]):
                assert all(child[0] not in inst_map for child in children)
                child_insts = {
                    child[0]: language.BoundVariable(child[0].name, instantiation[idx])
                    for idx, child in enumerate(children)}
                stack.append((remaining_subtrees, inst_map | child_insts))

        return instantiations

        # for variable_chain in variable_chains:
        #     nonterminal_sequences: Set[Tuple[str, ...]] = set([])
        #
        #     partial_sequences: Set[Tuple[str, ...]] = {("<start>",)}
        #     while partial_sequences:
        #         partial_sequence = next(iter(partial_sequences))
        #         partial_sequences.remove(partial_sequence)
        #         if len(partial_sequence) == len(variable_chain):
        #             nonterminal_sequences.add(tuple(reversed(partial_sequence)))
        #             continue
        #
        #         partial_sequences.update({
        #             partial_sequence + (to_nonterminal,)
        #             for to_nonterminal in reachable[partial_sequence[-1]]
        #             if to_nonterminal not in self.exclude_nonterminals})
        #
        #     new_instantiations: List[Dict[NonterminalPlaceholderVariable, language.BoundVariable]] = [
        #         {variable: language.BoundVariable(variable.name, nonterminal_sequence[idx])
        #          for idx, variable in enumerate(variable_chain[:-1])}
        #         for nonterminal_sequence in nonterminal_sequences]
        #
        #     if not instantiations:
        #         instantiations = new_instantiations
        #     else:
        #         previous_instantiations = instantiations
        #         instantiations = []
        #         for previous_instantiation in previous_instantiations:
        #             for new_instantiation in new_instantiations:
        #                 # NOTE: There might be clashes, since multiple chains are generated if there is
        #                 #       more than one variable in a match expression. We have to first check if
        #                 #       two instantiations happen to conform to each other.
        #                 if any(new_instantiation[key].n_type != previous_instantiation[key].n_type for key in
        #                        set(new_instantiation.keys()).intersection(set(previous_instantiation.keys()))):
        #                     continue
        #
        #                 instantiations.append(previous_instantiation | new_instantiation)
        #
        # return instantiations

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

                for mexpr_strings in self._infer_mexpr(in_nonterminal, nonterminal_types):
                    def replace_with_var(elem: str) -> str | language.Variable:
                        try:
                            return next(var for idx, var in enumerate(mexpr_ph.variables)
                                        if elem == var.n_type.replace(">", f"-{hash((var.n_type, idx))}>"))
                        except StopIteration:
                            return elem

                    mexpr_elements = [replace_with_var(element) for element in mexpr_strings]

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
            nonterminal_types: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        assert all(self.graph.reachable(in_nonterminal, target) for target in nonterminal_types)

        result: Set[Tuple[str, ...]] = set()

        candidate_trees: List[ParseTree] = [(in_nonterminal, None)]
        for i in range(self.mexpr_expansion_limit + 1):
            if not candidate_trees:
                break

            old_candidates = list(candidate_trees)
            candidate_trees = []
            for tree in old_candidates:
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

                nonterminal_occurrences: List[Tuple[Path, ...]] = []
                trie = trie_from_parse_tree(tree)
                stack: List[Tuple[Tuple[Path, ...], Tuple[str], str]] = \
                    [((), nonterminal_types, path_to_trie_key(()))]
                while stack:
                    matches, remaining_types, trie_key = stack.pop(0)
                    if not remaining_types:
                        nonterminal_occurrences.append(matches)

                    cur_path, cur_tree = trie[trie_key]

                    if cur_tree[0] == remaining_types[0]:
                        remaining_types = remaining_types[1:]
                        matches = matches + (cur_path,)

                        if not remaining_types:
                            nonterminal_occurrences.append(matches)
                            continue

                        # For the next trie key, skip the whole subtree of cur_path
                        if not cur_tree[1]:
                            next_key = next_trie_key(trie, trie_key)
                        else:
                            suffixes = list(filter(None, trie.suffixes(trie_key)))
                            assert suffixes
                            next_key = next_trie_key(trie, trie_key + suffixes[-1])
                    else:
                        next_key = next_trie_key(trie, trie_key)

                    if next_key is not None:
                        stack.append((matches, remaining_types, next_key))

                for matching_seq in nonterminal_occurrences:
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

                    result.add(tuple([tree[0] for _, tree in tree_leaves(bind_expr_tree)]))

                expanded_trees = expand_tree(tree, self.canonical_grammar)
                if self.max_nonterminals_in_mexpr is not None:
                    expanded_trees = [
                        t for t in expanded_trees
                        if (len([True for _, leaf in tree_leaves(t) if is_nonterminal(leaf[0])]) <=
                            self.max_nonterminals_in_mexpr)
                    ]
                candidate_trees.extend(expanded_trees)

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
            tries: List[datrie.Trie]) -> Set[language.Formula]:
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
                for trie in tries:
                    container_tries = [
                        get_subtrie(trie, path_key) for path_key, (path, subtree) in trie.items()
                        if subtree.value == container_nonterminal]

                    for container_trie in container_tries:
                        num_occs = len([
                            subtree for _, subtree in container_trie.values()
                            if subtree.value == elem_nonterminal])

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
            tries: List[datrie.Trie]):
        result: Set[language.Formula] = set([])
        string_placeholder_insts = self._get_string_placeholder_instantiations(inst_patterns, tries)

        for formula in string_placeholder_insts:
            if not string_placeholder_insts[formula]:
                result.add(formula)
                continue

            # We make instantiations for DSTRINGS placeholders map to *sets of tuples* of strings instead of
            # sets of strings. To prevent explosion, we apply a simple heuristic: We consider numbers and
            # single letters as irrelevant. This holds for our use cases (e.g., considering 'xmlns', 'sqrt',
            # and '*') but can be problematic in cases where something like 'f' should be considered. However,
            # some reduction *has* to be done, and this works for our use cases and seems to be sensile in
            # general (protected/pre-defined identifiers are rarely numbers or single characters).
            insts = {
                ph:
                    s if isinstance(ph, StringPlaceholderVariable)
                    else (s_ := {elem for elem in s if not is_int(elem) and elem not in string.ascii_letters},
                          {c for k in range(1, len(s_) + 1) for c in tuple(itertools.combinations(s_, k))})[-1]
                for ph, s in string_placeholder_insts[formula].items()}

            string_ph_insts: List[Dict[StringPlaceholderVariable, str | Set[str]]] = \
                dict_of_lists_to_list_of_dicts(insts)

            for instantiation in string_ph_insts:
                # TODO: We have to account for sets in instantiations in the abstract
                #       evaluation, or expand the formula before. Probably try the former.
                if any(
                        not approximately_evaluate_abst_for(
                            formula,
                            self.grammar,
                            self.graph,
                            {language.Constant("start", "<start>"): trie[path_to_trie_key(())]} | instantiation,
                            trie
                        ).is_false()
                        for trie in tries):
                    instantiated_formula = formula
                    for ph, inst in instantiation.items():
                        instantiated_formula = language.replace_formula(
                            instantiated_formula,
                            functools.partial(substitute_string_placeholder, ph, inst))
                    result.add(instantiated_formula)

        return result

    def _get_string_placeholder_instantiations(
            self,
            inst_patterns: Set[language.Formula],
            tries: List[datrie.Trie]) -> Dict[language.Formula, Dict[StringPlaceholderVariable, Set[str]]]:
        if all(not isinstance(placeholder, StringPlaceholderVariableTypes)
               for inst_pattern in inst_patterns
               for placeholder in get_placeholders(inst_pattern)):
            return dict.fromkeys(inst_patterns)

        # TODO: Record fragments inside their *contexts*, and only instantiate in the right context.
        #       An <ID> in at the place of a function name is different than an <ID> at the place of
        #       a variable.
        fragments: Dict[str, Set[str]] = {nonterminal: set([]) for nonterminal in self.grammar}
        for trie in tries:
            remaining_paths: List[Tuple[Path, language.DerivationTree]] = list(
                sorted(list(trie.values()), key=lambda p: len(p[0])))

            handled_paths: Dict[str, Set[Path]] = {nonterminal: set([]) for nonterminal in self.grammar}
            while remaining_paths:
                path, tree = remaining_paths.pop(0)
                if not is_nonterminal(tree.value):
                    continue

                tree_string = str(tree)
                if not tree_string:
                    continue

                # NOTE: We exclude substrings from fragments; e.g., if we have a <digits>
                #       "1234", don't include the <digits> "34". This might lead
                #       to imprecision, but otherwise the search space tends to explode.
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

        result: Dict[language.Formula, Dict[StringPlaceholderVariableTypes, Set[str]]] = {}
        for inst_pattern in inst_patterns:
            def extract_instantiations(subformula: language.Formula) -> None:
                nonlocal result

                if not any(isinstance(arg, StringPlaceholderVariableTypes) for arg in subformula.free_variables()):
                    return

                non_ph_vars = {v for v in subformula.free_variables() if not isinstance(v, PlaceholderVariable)}

                insts: Set[str] = set(functools.reduce(
                    set.__or__, [fragments.get(var.n_type, set([])) for var in non_ph_vars]))

                ph_vars = {v for v in subformula.free_variables() if isinstance(v, StringPlaceholderVariableTypes)}
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
        inst: str | tuple[str, ...],
        formula: language.Formula) -> language.Formula | bool:
    # A tuple of instantiations indicates that a disjunction of formulas shall be
    # returned, including one instantiated formula for each instantiation.

    if not any(v == ph for v in formula.free_variables()
               if isinstance(v, StringPlaceholderVariableTypes)):
        return False

    def perform_single_inst(inst_str: str) -> Optional[language.Formula]:
        if isinstance(formula, language.SMTFormula):
            return language.SMTFormula(cast(z3.BoolRef, z3_subst(formula.formula, {
                ph.to_smt(): z3.StringVal(inst_str)
            })), *[v for v in formula.free_variables() if v != ph])
        elif isinstance(formula, language.StructuralPredicateFormula):
            return language.StructuralPredicateFormula(
                formula.predicate,
                *[inst_str if arg == ph else arg for arg in formula.args]
            )
        elif isinstance(formula, language.SemanticPredicateFormula):
            return language.SemanticPredicateFormula(
                formula.predicate,
                *[inst_str if arg == ph else arg for arg in formula.args])

        return None

    if isinstance(inst, str):
        inst = inst,

    single_insts = list(map(perform_single_inst, inst))
    if any(single_inst is None for single_inst in single_insts):
        return False

    return functools.reduce(language.Formula.__or__, single_insts)


def approximately_evaluate_abst_for(
        formula: language.Formula,
        grammar: Grammar,
        graph: gg.GrammarGraph,
        assignments: Dict[language.Variable, Tuple[Path, language.DerivationTree] | str | Set[str]],
        trie: Optional[datrie.Trie] = None) -> ThreeValuedTruth:
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
                for var in formula.free_variables()
            }

            # Expand multiple instantiations for `DisjunctiveStringsPlaceholderVariable`s
            split_assignments: List[Dict[language.Variable, Tuple[Path, language.DerivationTree] | str]] = [
                dict(t) for t in itertools.product(*[
                    ((var, val),) if not isinstance(var, DisjunctiveStringsPlaceholderVariable)
                    else tuple(itertools.product((var,), e_assert(val, lambda v: isinstance(v, tuple))))
                    for var, val in assignments.items()
                    if var in formula.free_variables()])]

            # The multiple assignments in the list are treated as a disjunction:
            # As soon as any assignment yields a `True` result, we return `True`.

            for single_assignment in split_assignments:
                args_instantiation = ()
                for arg in translation[0]:
                    var = var_map[arg]
                    if isinstance(var, StringPlaceholderVariableTypes):
                        args_instantiation += (single_assignment[var],)
                    else:
                        args_instantiation += (str(single_assignment[var][1]),)

                if translation[1](args_instantiation) if args_instantiation else translation[1]:
                    return ThreeValuedTruth.true()

            return ThreeValuedTruth.false()
        except DomainError:
            return ThreeValuedTruth.false()
        except NotImplementedError:
            return is_valid(z3.substitute(
                formula.formula,
                *tuple({z3.String(symbol.name): z3.StringVal(str(symbol_assignment[1]))
                        for symbol, symbol_assignment
                        in assignments.items()}.items())))

    elif isinstance(formula, language.NumericQuantifiedFormula):
        return approximately_evaluate_abst_for(formula.inner_formula, grammar, graph, assignments, trie)
    elif isinstance(formula, language.QuantifiedFormula):
        assert isinstance(formula.in_variable, language.Variable)
        assert formula.in_variable in assignments
        in_path, in_inst = assignments[formula.in_variable]

        if formula.bind_expression is None:
            sub_trie = get_subtrie(trie, in_path)

            new_assignments: List[Dict[language.Variable, Tuple[Path, language.DerivationTree]]] = []
            for path, subtree in sub_trie.values():
                if subtree.value == formula.bound_variable.n_type:
                    new_assignments.append({formula.bound_variable: (in_path + path, subtree)})
        elif isinstance(formula.bind_expression.bound_elements[0], MexprPlaceholderVariable):
            mexpr_placeholder = cast(MexprPlaceholderVariable, formula.bind_expression.bound_elements[0])

            # First, get all matches for the bound variables
            new_assignments: List[Dict[language.Variable, Tuple[Path, language.DerivationTree]]] = []
            for path, subtree in get_subtrie(trie, in_path).values():
                if subtree.value == formula.bound_variable.n_type:
                    new_assignments.append({formula.bound_variable: (in_path + path, subtree)})

            # Next, find subtrees below those matches matching the mexpr placeholder in the correct order
            for idx, placeholder_variable in enumerate(mexpr_placeholder.variables):
                if not idx:
                    for assignment in list(new_assignments):
                        new_assignments.remove(assignment)

                        sub_trie = get_subtrie(trie, assignment[formula.bound_variable][0])
                        matches = [
                            (in_path + path, subtree)
                            for path, subtree in sub_trie.values()
                            if subtree.value == placeholder_variable.n_type]
                        if not matches:
                            continue

                        new_assignments.extend([
                            assignment | {placeholder_variable: (path, subtree)}
                            for path, subtree in matches
                        ])
                else:
                    last_variable: NonterminalPlaceholderVariable = mexpr_placeholder.variables[idx - 1]
                    for assignment in list(new_assignments):
                        new_assignments.remove(assignment)

                        sub_trie = get_subtrie(trie, assignment[formula.bound_variable][0])
                        last_path: Path = assignment[last_variable][0]
                        for sub_path_key in sub_trie.keys(path_to_trie_key(last_path)):
                            del sub_trie[sub_path_key]

                        next_key = next_trie_key(sub_trie, path_to_trie_key(last_path))
                        while next_key is not None:
                            sub_sub_trie = get_subtrie(sub_trie, next_key)

                            matches = [
                                (last_path + path, subtree)
                                for path, subtree in sub_sub_trie.values()
                                if subtree.value == placeholder_variable.n_type]
                            if not matches:
                                continue

                            new_assignments.extend([
                                assignment | {placeholder_variable: (path, subtree)}
                                for path, subtree in matches
                            ])

                            for sub_path_key in sub_trie.keys(next_key):
                                del sub_trie[sub_path_key]

                            next_key = next_trie_key(sub_trie, path_to_trie_key(last_path))

            # For universal formulas, we only consider the first 3 new assignments
            # to save time. After match expression placeholders are instantiated,
            # we have the chance for a more precise check.
            assert all(formula.bound_variable in assignment for assignment in new_assignments)
            assert all(var in assignment for assignment in new_assignments for var in mexpr_placeholder.variables)
            if isinstance(formula, language.ForallFormula):
                new_assignments = new_assignments[:3]
        else:
            new_assignments = [
                {var: (in_path + path, tree) for var, (path, tree) in new_assignment.items()}
                for new_assignment in matches_for_quantified_formula(
                    formula, grammar, in_inst, {})]

        if not new_assignments:
            return ThreeValuedTruth.false()

        new_assignments = [
            new_assignment | assignments
            for new_assignment in new_assignments]

        if isinstance(formula, language.ExistsFormula):
            return ThreeValuedTruth.from_bool(any(
                not approximately_evaluate_abst_for(
                    formula.inner_formula, grammar, graph, new_assignment, trie).is_false()
                for new_assignment in new_assignments))
        else:
            return ThreeValuedTruth.from_bool(all(
                not approximately_evaluate_abst_for(
                    formula.inner_formula, grammar, graph, new_assignment, trie).is_false()
                for new_assignment in new_assignments))
    elif isinstance(formula, language.StructuralPredicateFormula):
        if any(isinstance(arg, PlaceholderVariable) for arg in formula.args):
            return ThreeValuedTruth.unknown()

        arg_insts = [
            arg if isinstance(arg, str)
            else next(path for path, subtree in trie.values() if subtree.id == arg.id)
            if isinstance(arg, language.DerivationTree)
            else assignments[arg][0]
            for arg in formula.args]

        return ThreeValuedTruth.from_bool(formula.predicate.evaluate(trie.get(path_to_trie_key(()))[1], *arg_insts))
    elif isinstance(formula, language.SemanticPredicateFormula):
        if any(isinstance(arg, PlaceholderVariable) for arg in formula.args):
            return ThreeValuedTruth.unknown()

        arg_insts = [arg if isinstance(arg, language.DerivationTree) or arg not in assignments
                     else assignments[arg][1]
                     for arg in formula.args]
        eval_res = formula.predicate.evaluate(graph, *arg_insts)

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
        return ThreeValuedTruth.not_(
            approximately_evaluate_abst_for(formula.args[0], grammar, graph, assignments, trie))
    elif isinstance(formula, language.ConjunctiveFormula):
        # Relaxation: Unknown is OK, only False is excluded.
        return ThreeValuedTruth.from_bool(all(
            not approximately_evaluate_abst_for(sub_formula, grammar, graph, assignments, trie).is_false()
            for sub_formula in formula.args))
    elif isinstance(formula, language.DisjunctiveFormula):
        return ThreeValuedTruth.from_bool(any(
            not approximately_evaluate_abst_for(sub_formula, grammar, graph, assignments, trie).is_false()
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

    def predicate(self, formula: language.Formula, tries: List[datrie.Trie]) -> bool:
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

    def predicate(self, formula: language.Formula, _: List[datrie.Trie]) -> bool:
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
        StringPlaceholderVariable,
        DisjunctiveStringsPlaceholderVariable,
    }

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

        return all_patterns - exclude

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
