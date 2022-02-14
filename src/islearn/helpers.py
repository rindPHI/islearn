import itertools
from typing import Callable, TypeVar, Optional, Iterable, Tuple, Set, List, Dict

from pathos import multiprocessing as pmp

S = TypeVar("S")
T = TypeVar("T")


def e_assert_present(expression: T, message: Optional[str] = None) -> T:
    return e_assert(expression, lambda e: e is not None, message)


def e_assert(expression: T, assertion: Callable[[T], bool], message: Optional[str] = None) -> T:
    assert assertion(expression), message or ""
    return expression


def parallel_all(f: Callable[[T], bool], iterable: Iterable[T], chunk_size=16, processes=pmp.cpu_count()) -> bool:
    l = list(iterable)
    chunked_list = [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]

    idx = 0
    while idx < len(chunked_list):
        with pmp.ProcessingPool(processes=processes) as pool:
            if not all(pool.map(lambda chunk: all(f(elem) for elem in chunk), chunked_list[idx:idx + processes])):
                return False

        idx += processes

    return True


def parallel_any(f: Callable[[T], bool], iterable: Iterable[T], chunk_size=16, processes=pmp.cpu_count()) -> bool:
    l = list(iterable)
    chunked_list = [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]

    idx = 0
    while idx < len(chunked_list):
        with pmp.ProcessingPool(processes=processes) as pool:
            if any(pool.map(lambda chunk: any(f(elem) for elem in chunk), chunked_list[idx:idx + processes])):
                return True

        idx += processes

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
