import itertools
import math
from functools import reduce
from typing import Callable, TypeVar, Optional, Iterable, Tuple, Set, List, Dict, Sequence

import isla.language
import z3
from fuzzingbook.Parser import canonical
from isla.helpers import is_nonterminal, dict_of_lists_to_list_of_dicts
from isla.language import DerivationTree
from isla.type_defs import Path, Grammar
from pathos import multiprocessing as pmp

S = TypeVar("S")
T = TypeVar("T")


def e_assert_present(expression: T, message: Optional[str] = None) -> T:
    return e_assert(expression, lambda e: e is not None, message)


def e_assert(expression: T, assertion: Callable[[T], bool], message: Optional[str] = None) -> T:
    assert assertion(expression), message or ""
    return expression


def parallel_all(f: Callable[[T], bool], iterable: Iterable[T], chunk_size=16, processes=pmp.cpu_count()) -> bool:
    # l = list(iterable)
    # chunked_list = [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]
    #
    # idx = 0
    # while idx < len(chunked_list):
    #     with pmp.ProcessingPool(processes=processes) as pool:
    #         if not all(pool.map(lambda chunk: all(f(elem) for elem in chunk), chunked_list[idx:idx + processes])):
    #             return False
    #
    #     idx += processes
    #
    # return True
    with pmp.ProcessingPool(processes=processes) as pool:
        results = pool.uimap(f, iterable)
        for result in results:
            if not result:
                return False

    return True


def parallel_any(f: Callable[[T], bool], iterable: Iterable[T], processes=pmp.cpu_count()) -> bool:
    with pmp.ProcessingPool(processes=processes) as pool:
        results = pool.uimap(f, iterable)
        for result in results:
            if result:
                return True

    return False


def mappings(a: Iterable[S], b: Iterable[T]) -> List[Dict[S, T]]:
    a = list(a)
    return [
        dict(zip(a, item))
        for item in itertools.product(b, repeat=len(a))]


def connected_chains(relation: Iterable[Tuple[S, S]]) -> Set[Tuple[S, ...]]:
    result: Set[Tuple[S, ...]] = set(relation)
    while True:
        matches: Set[Tuple[Tuple[S, ...], Tuple[S, ...]]] = {
            (chain_1, chain_2)
            for chain_1 in result
            for chain_2 in result
            if chain_1[-1] == chain_2[0]
        }

        new_chains: Set[Tuple[S, ...]] = {
            chain_1 + chain_2[1:]
            for chain_1, chain_2 in matches
        }

        result_until_now = result | new_chains

        if result_until_now == result:
            break

        result = result_until_now

    # Exclude subsequences. Cannot do this before, since we need the original
    # chains for creating longer ones.
    return {
        chain for idx, chain in enumerate(result)
        if not any(
            idx != other_idx and
            len(other_chain) > len(chain) and
            any(
                other_chain[idx_1:idx_2] == chain
                for idx_1 in range(len(other_chain))
                for idx_2 in range(idx_1, len(other_chain) + 1)
            )
            for other_idx, other_chain in enumerate(result)
        )
    }


def transitive_closure(relation: Iterable[Tuple[S, T]]) -> Set[Tuple[S, T]]:
    closure = set(relation)
    while True:
        new_relations: Set[Tuple[S, T]] = {
            (x, w)
            for x, y in closure
            for q, w in closure if q == y}

        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break

        closure = closure_until_now

    return closure


def non_consecutive_ordered_sub_sequences(sequence: Iterable[T], length: int) -> Set[Tuple[T, ...]]:
    sequence = list(sequence)
    if len(sequence) < length:
        return set()

    filters_list = [f for f in list(itertools.product([0, 1], repeat=len(sequence))) if sum(f) == length]

    return {tuple(itertools.compress(sequence, a_filter)) for a_filter in filters_list}


def all_interleavings(a: Sequence[S], b: Sequence[T]) -> List[List[S | T]]:
    slots = [None] * (len(a) + len(b))
    for splice in itertools.combinations(range(0, len(slots)), len(b)):
        it_B = iter(b)
        for s in splice:
            slots[s] = next(it_B)
        it_A = iter(a)
        slots = [e if e else next(it_A) for e in slots]
        yield slots
        slots = [None] * (len(slots))

    return slots


def construct_multiple(constructor: Callable[..., T], args: List[Iterable]) -> Set[T]:
    """
    :param constructor: Constructor to apply to individual arguments.
    :param args: List of different argument options. Length of outer list must match cardinality of constructor.
    :return: Instantiated objects for the different possibilities.
    """

    return {constructor(*combination) for combination in itertools.product(*args)}


def reduce_multiple(function: Callable[[T], T], sequence: List[Iterable[T]]) -> Set[T]:
    return {reduce(function, combination) for combination in itertools.product(*sequence)}


def replace_formula_by_formulas(
        in_formula: isla.language.Formula,
        to_replace: Callable[[isla.language.Formula], Optional[Iterable[isla.language.Formula]]]
) -> Set[isla.language.Formula]:
    """
    Replaces a formula inside a conjunction or disjunction.
    to_replace is either (1) a formula to replace, or (2) a predicate which holds if the given formula
    should been replaced (if it returns True, replace_with must not be None), or (3) a function returning
    the formula to replace if the subformula should be replaced, or False otherwise. For (3), replace_with
    may be None (it is irrelevant).
    """

    maybe_results = to_replace(in_formula)
    if maybe_results is not None:
        return {r for result in maybe_results for r in replace_formula_by_formulas(result, to_replace)}

    if isinstance(in_formula, isla.language.ConjunctiveFormula):
        return reduce_multiple(lambda a, b: a & b, [
            replace_formula_by_formulas(child, to_replace)
            for child in in_formula.args])
    elif isinstance(in_formula, isla.language.DisjunctiveFormula):
        return reduce_multiple(lambda a, b: a | b, [
            replace_formula_by_formulas(child, to_replace)
            for child in in_formula.args])
    elif isinstance(in_formula, isla.language.NegatedFormula):
        return {
            -child_result
            for child_result in replace_formula_by_formulas(in_formula.args[0], to_replace)
        }
    elif isinstance(in_formula, isla.language.ForallFormula):
        in_formula: isla.language.ForallFormula
        return construct_multiple(
            lambda inner_formula: isla.language.ForallFormula(
                in_formula.bound_variable,
                in_formula.in_variable,
                inner_formula,
                in_formula.bind_expression,
                in_formula.already_matched,
                id=in_formula.id
            ), [replace_formula_by_formulas(in_formula.inner_formula, to_replace)])
    elif isinstance(in_formula, isla.language.ExistsFormula):
        in_formula: isla.language.ExistsFormula
        return construct_multiple(
            lambda inner_formula: isla.language.ExistsFormula(
                in_formula.bound_variable,
                in_formula.in_variable,
                inner_formula,
                in_formula.bind_expression),
            [replace_formula_by_formulas(in_formula.inner_formula, to_replace)])
    elif isinstance(in_formula, isla.language.ExistsIntFormula):
        in_formula: isla.language.ExistsIntFormula
        return construct_multiple(
            lambda inner_formula:
            isla.language.ExistsIntFormula(
                in_formula.bound_variable,
                inner_formula),
            [replace_formula_by_formulas(in_formula.inner_formula, to_replace)])
    elif isinstance(in_formula, isla.language.ForallIntFormula):
        in_formula: isla.language.ForallIntFormula
        return construct_multiple(
            lambda inner_formula:
            isla.language.ForallIntFormula(
                in_formula.bound_variable,
                inner_formula),
            [replace_formula_by_formulas(in_formula.inner_formula, to_replace)])

    return {in_formula}


def expand_tree(
        tree: DerivationTree,
        grammar: Grammar,
        limit: Optional[int] = None) -> List[DerivationTree]:
    canonical_grammar = canonical(grammar)

    nonterminal_expansions = {
        leaf_path: [
            [DerivationTree(child, None if is_nonterminal(child) else [], precompute_is_open=False)
             for child in expansion]
            for expansion in canonical_grammar[leaf_node.value]
        ]
        for leaf_path, leaf_node in tree.open_leaves()
    }

    possible_expansions: List[Dict[Path, List[DerivationTree]]] = \
        dict_of_lists_to_list_of_dicts(nonterminal_expansions)

    assert len(possible_expansions) == math.prod(len(values) for values in nonterminal_expansions.values())

    if len(possible_expansions) == 1 and not possible_expansions[0]:
        return []

    if limit:
        possible_expansions = possible_expansions[:limit]

    result: List[DerivationTree] = []
    for possible_expansion in possible_expansions:
        expanded_tree = tree
        for path, new_children in possible_expansion.items():
            leaf_node = expanded_tree.get_subtree(path)
            expanded_tree = expanded_tree.replace_path(
                path, DerivationTree(leaf_node.value, new_children, leaf_node.id, precompute_is_open=False))

        result.append(expanded_tree)

    assert not limit or len(result) <= limit
    return result
