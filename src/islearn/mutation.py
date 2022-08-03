import logging
import random
from typing import Set, Dict, Callable, Tuple, Iterable, Generator, Optional

from grammar_graph import gg
from isla.existential_helpers import paths_between, path_to_tree
from isla.fuzzer import GrammarCoverageFuzzer
from isla.helpers import is_nonterminal, canonical
from isla.language import DerivationTree
from isla.type_defs import Grammar, Path

random = random.SystemRandom()

class MutationFuzzer:
    """
    A k-path coverage guided, grammar-aware mutation fuzzer using input fragments.
    """

    def __init__(
            self,
            grammar: Grammar,
            seed: Iterable[DerivationTree],
            property: Callable[[DerivationTree], bool] = lambda tree: True,
            k: int = 3,
            min_mutations: int = 2,
            max_mutations: int = 10):
        self.logger = logging.getLogger(type(self).__name__)
        self.grammar = grammar
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.seed = set(seed)
        self.property = property
        self.k = k
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations

        self.fuzzer = GrammarCoverageFuzzer(grammar)

        self.population = set([])
        self.coverages_seen: Set[Tuple[gg.Node, ...]] = set([])
        self.fragments: Dict[str, Set[DerivationTree]] = {}

        self.reset()

        self.mutators: Dict[Callable[[DerivationTree, Path], DerivationTree | None]] = {
            self.replace_with_fragment: 4,
            self.swap_subtrees: 3,
            self.replace_with_random_subtree: 2,
            self.generalize: 1,
        }

    def reset(self):
        self.population = set(self.seed)
        self.coverages_seen = set([])
        self.coverages_seen = {p for inp in self.population for p in self.coverages_of(inp)}

        for sample in self.population:
            self.update_fragments(sample)

    def update_fragments(self, sample: DerivationTree) -> None:
        for _, tree in sample.paths():
            self.fragments.setdefault(tree.value, set([])).add(tree)

    def coverages_of(self, inp: DerivationTree) -> Set[Tuple[gg.Node, ...]]:
        return self.graph.k_paths_in_tree(inp.to_parse_tree(), self.k)

    def fuzz(self) -> DerivationTree:
        num_mutations = random.randint(self.max_mutations, self.max_mutations)
        curr_inp = random.choice(list(self.population))
        mutations = 0
        while mutations < num_mutations:
            maybe_result = self.mutate(curr_inp)
            if maybe_result is not None:
                curr_inp = maybe_result
                mutations += 1
        return curr_inp

    def mutate(self, inp: DerivationTree) -> DerivationTree | None:
        paths = [path for path, subtree in inp.paths() if is_nonterminal(subtree.value)]
        while paths:
            path = random.choice(paths)
            paths.remove(path)

            mutators = dict(self.mutators)
            while mutators:
                mutator = random.choices(
                    list(mutators.keys()),
                    list(mutators.values()), k=1)[0]
                del mutators[mutator]

                result = mutator(inp, path)
                if result is not None:
                    return result

        return None

    def replace_with_fragment(self, inp: DerivationTree, path: Path) -> DerivationTree | None:
        subtree = inp.get_subtree(path)
        if not is_nonterminal(subtree.value):
            return None

        different_fragments = [
            fragment for fragment in self.fragments.get(subtree.value, [])
            if fragment and not fragment.structurally_equal(subtree)
        ]

        if not different_fragments:
            return inp

        result = inp.replace_path(path, random.choice(different_fragments))
        assert result is None or self.graph.tree_is_valid(result.to_parse_tree())
        return result

    def replace_with_random_subtree(self, inp: DerivationTree, path: Path) -> DerivationTree | None:
        subtree = inp.get_subtree(path)
        if not is_nonterminal(subtree.value):
            return None

        result = inp.replace_path(path, self.fuzzer.expand_tree(DerivationTree(subtree.value, None)))
        assert result is None or self.graph.tree_is_valid(result.to_parse_tree())
        return result

    def swap_subtrees(self, inp: DerivationTree, path: Path) -> DerivationTree | None:
        subtree = inp.get_subtree(path)
        if not is_nonterminal(subtree.value):
            return None

        matches = [tree for other_path, tree in inp.filter(lambda t: t.value == subtree.value)
                   if other_path != path]

        if not matches:
            return None

        result = inp.replace_path(path, random.choice(matches))
        assert result is None or self.graph.tree_is_valid(result.to_parse_tree())
        return result

    def generalize(self, inp: DerivationTree, path: Path) -> DerivationTree | None:
        subtree = inp.get_subtree(path)
        if not is_nonterminal(subtree.value):
            return None

        self_embedding_trees = [
            self_embedding_tree
            for self_embedding_path in paths_between(self.graph, subtree.value, subtree.value)
            for self_embedding_tree in path_to_tree(canonical(self.grammar), self_embedding_path)]

        if not self_embedding_trees:
            return None

        self_embedding_tree = random.choice(self_embedding_trees)
        matching_leaf = random.choice([p for p, t in self_embedding_tree.leaves() if t.value == subtree.value])

        result = inp.replace_path(path, self_embedding_tree.replace_path(matching_leaf, subtree))
        result = self.expand_tree(result)
        assert result is None or self.graph.tree_is_valid(result.to_parse_tree())

        return result

    def expand_tree(self, inp: DerivationTree) -> DerivationTree:
        for path, tree in inp.open_leaves():
            if random.random() < .3 or tree.value not in self.fragments:
                inp = inp.replace_path(path, self.fuzzer.expand_tree(tree))
            else:
                inp = inp.replace_path(path, random.choice(list(self.fragments[tree.value])))

        assert inp.is_complete()
        return inp

    def run(
            self,
            num_iterations: Optional[int] = 500,
            alpha: float = 0.1,
            extend_fragments: bool = True,
            yield_negative=False) -> Generator[DerivationTree, None, None]:
        unsuccessful_tries = 0

        i = 0
        while True:
            if num_iterations is not None and i >= num_iterations:
                break

            curr_alpha = 1 - (unsuccessful_tries / (i + 1))
            if curr_alpha < alpha:
                if i * 10 > (num_iterations or 500):
                    break

            inp = self.fuzz()

            if self.process_new_input(inp, extend_fragments):
                yield inp
            else:
                unsuccessful_tries += 1
                if yield_negative:
                    yield inp
                self.logger.debug("current alpha: %f, threshold: %f", curr_alpha, alpha)

            i += 1

    def process_new_input(self, inp: DerivationTree, extend_fragments: bool = True) -> bool:
        new_coverage = self.coverages_seen - self.coverages_of(inp)
        if inp in self.population or not self.property(inp) or not new_coverage:
            return False

        self.coverages_seen.update(new_coverage)
        self.population.add(inp)
        if extend_fragments:
            self.update_fragments(inp)

        return True
