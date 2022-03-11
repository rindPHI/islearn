import logging
from typing import Callable, Set

from grammar_graph import gg
from isla.fuzzer import GrammarFuzzer
from isla.helpers import is_nonterminal
from isla.language import DerivationTree
from isla.type_defs import Grammar, Path
from pathos import multiprocessing as pmp


class InputReducer:
    def __init__(self, grammar: Grammar, property: Callable[[DerivationTree], bool], k: int = 3):
        self.grammar = grammar
        self.graph = gg.GrammarGraph.from_grammar(grammar)
        self.property = property
        self.logger = logging.getLogger(__name__)
        self.k = k

    def reduce_by_smallest_subtree_replacement(self, inp: DerivationTree) -> DerivationTree:
        self.logger.debug("Reducing %s", inp)

        k_paths_in_inp = {
            path for path in inp.k_paths(self.graph, k=self.k)
            if not isinstance(path[-1], gg.TerminalNode) or
               (not isinstance(path[-1], gg.TerminalNode) and len(path[-1].symbol) > 1)}

        result = inp
        while True:
            def substitute_action(path: Path, _: DerivationTree):
                nonlocal result

                if not result.is_valid_path(path):
                    return

                subtree = result.get_subtree(path)

                matches = [t for p, t in subtree.filter(lambda t: t.value == subtree.value) if p]
                if not matches:
                    return

                matches = sorted(matches, key=DerivationTree.__len__)

                for match in matches:
                    # Don't replace subtree by match if it contains a k-path that will vanish otherwise
                    potential_replacement = result.replace_path(path, match)

                    k_paths_in_replacement = {
                        path for path in potential_replacement.k_paths(self.graph, k=self.k)
                        if (not isinstance(path[-1], gg.TerminalNode) or
                            (not isinstance(path[-1], gg.TerminalNode) and len(path[-1].symbol) > 1))}

                    if any(path not in k_paths_in_replacement for path in k_paths_in_inp):
                        continue

                    # self.logger.debug("Replacing %s with %s", subtree, match)

                    if self.property(potential_replacement):
                        result = potential_replacement
                        return

            # inp.traverse(substitute_action, kind=DerivationTree.TRAVERSE_PREORDER)
            inp.bfs(substitute_action)

            if result == inp:
                break

            inp = result

        self.logger.debug("Result: %s", result)
        return result

    def reduce_by_abstraction(self, inp: DerivationTree) -> DerivationTree:
        def can_abstract(tree: DerivationTree, path: Path) -> bool:
            subtree = tree.get_subtree(path)

            assert is_nonterminal(subtree.value)

            # for _ in range(10):
            #     new_leaf_tree = fuzzer.expand_tree(DerivationTree(subtree.value, None))
            #     new_tree = tree.replace_path(path, new_leaf_tree)
            #     if not self.property(new_tree):
            #         return False
            #
            # return True

            def check(_):
                new_leaf_tree = GrammarFuzzer(self.grammar).expand_tree(DerivationTree(subtree.value, None))
                new_tree = tree.replace_path(path, new_leaf_tree)
                if not self.property(new_tree):
                    return False
                return True

            with pmp.ProcessingPool(processes=pmp.cpu_count()) as pool:
                for result in pool.uimap(check, range(10)):
                    if not result:
                        return False

                return True

        def reduce_action(path: Path, subtree: DerivationTree):
            nonlocal result
            if not is_nonterminal(subtree.value):
                return

            if path in do_not_abstract:
                return

            if can_abstract(result, path):
                result = result.replace_path(path, DerivationTree(subtree.value, None))
                return
            else:
                do_not_abstract.update({path[:idx] for idx in range(len(path))})

        result = inp
        do_not_abstract: Set[Path] = set([])
        result.traverse(reduce_action, kind=DerivationTree.TRAVERSE_POSTORDER)

        return result
