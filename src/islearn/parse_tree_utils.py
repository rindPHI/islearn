import logging
import math
from typing import List, Callable, Tuple, Generator, Optional

from isla.helpers import is_nonterminal
from isla.type_defs import ParseTree, Path, CanonicalGrammar


def get_subtree(tree: ParseTree, path: Path) -> ParseTree:
    """Access a subtree based on `path` (a list of children numbers)"""
    curr_node = tree
    while path:
        curr_node = curr_node[1][path[0]]
        path = path[1:]

    return curr_node


def replace_path(
        tree: ParseTree,
        path: Path,
        replacement_tree: ParseTree) -> ParseTree:
    """Returns tree where replacement_tree has been inserted at `path` instead of the original subtree"""
    stack: List[ParseTree] = [tree]
    for path_elem in path:
        stack.append(stack[-1][1][path_elem])

    stack[-1] = replacement_tree

    for path_elem in reversed(path):
        assert len(stack) > 1
        replacement = stack.pop()
        parent = stack.pop()

        node, children = parent
        new_children = children[:path_elem] + [replacement] + children[path_elem + 1:]

        stack.append((node, new_children))

    assert len(stack) == 1
    return stack[0]


def filter_tree(
        tree: ParseTree,
        f: Callable[[ParseTree], bool],
        enforce_unique: bool = False) -> List[Tuple[Path, ParseTree]]:
    result: List[Tuple[Path, ParseTree]] = []

    for path, subtree in tree_paths(tree):
        if f(subtree):
            result.append((path, subtree))

            if enforce_unique and len(result) > 1:
                raise RuntimeError(f"Found searched-for element more than once in {tree}")

    return result


TRAVERSE_PREORDER = 0
TRAVERSE_POSTORDER = 1


def traverse_tree(
        tree: ParseTree,
        action: Callable[[Path, ParseTree], None],
        abort_condition: Callable[[Path, ParseTree], bool] = lambda p, n: False,
        kind: int = TRAVERSE_PREORDER,
        reverse: bool = False) -> None:
    stack_1: List[Tuple[Path, ParseTree]] = [((), tree)]
    stack_2: List[Tuple[Path, ParseTree]] = []

    if kind == TRAVERSE_PREORDER:
        reverse = not reverse

    while stack_1:
        path, node = stack_1.pop()

        if abort_condition(path, node):
            return

        if kind == TRAVERSE_POSTORDER:
            stack_2.append((path, node))

        if kind == TRAVERSE_PREORDER:
            action(path, node)

        if node[1]:
            iterator = reversed(node[1]) if reverse else iter(node[1])

            for idx, child in enumerate(iterator):
                new_path = path + ((len(node[1]) - idx - 1) if reverse else idx,)
                stack_1.append((new_path, child))

    if kind == TRAVERSE_POSTORDER:
        while stack_2:
            action(*stack_2.pop())


def tree_paths(tree: ParseTree) -> List[Tuple[Path, ParseTree]]:
    def action(path, node):
        result.append((path, node))

    result: List[Tuple[Path, ParseTree]] = []
    traverse_tree(tree, action, kind=TRAVERSE_PREORDER)
    return result


def tree_to_string(tree: ParseTree, show_open_leaves: bool = False) -> str:
    result = []
    stack = [tree]

    while stack:
        node = stack.pop(0)
        symbol, children = node

        if not children:
            if children is not None:
                result.append("" if is_nonterminal(symbol) else symbol)
            else:
                result.append(symbol if show_open_leaves else "")

            continue

        stack = list(children) + stack

    return ''.join(result)


def open_leaves(tree: ParseTree) -> Generator[Tuple[Path, ParseTree], None, None]:
    return ((path, sub_tree) for path, sub_tree in tree_paths(tree) if sub_tree[1] is None)


def tree_leaves(tree: ParseTree) -> Generator[Tuple[Path, ParseTree], None, None]:
    return ((path, sub_tree) for path, sub_tree in tree_paths(tree) if not sub_tree[1])


def expand_tree(
        tree: ParseTree,
        canonical_grammar: CanonicalGrammar,
        limit: Optional[int] = None,
        expand_beyond_nonterminals: bool = False) -> List[ParseTree]:
    nonterminal_expansions = {
        (leaf_path, leaf_node): [
            [(child, None if is_nonterminal(child) else [])
             for child in expansion]
            for expansion in canonical_grammar[leaf_node[0]]
            if expand_beyond_nonterminals or any(is_nonterminal(child) for child in expansion)
        ]
        for leaf_path, leaf_node in open_leaves(tree)
    }

    result: List[ParseTree] = [tree]
    for (leaf_path, leaf_node), expansions in nonterminal_expansions.items():
        previous_result = result
        result = []
        for t in previous_result:
            for expansion in expansions:
                result.append(replace_path(
                    t,
                    leaf_path,
                    (leaf_node[0], expansion)))

                if limit and len(result) >= limit:
                    break
            if limit and len(result) >= limit:
                break
        if limit and len(result) >= limit:
            break

    assert ((limit and len(result) == limit) or
            len(result) == math.prod(len(values) for values in nonterminal_expansions.values()))

    return result
