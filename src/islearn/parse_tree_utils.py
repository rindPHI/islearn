import logging
import math
from typing import List, Callable, Tuple, Generator, Optional, Any, Sequence, TypeVar, Dict, cast

import datrie
from isla.helpers import is_nonterminal
from isla.trie import path_to_trie_key
from isla.type_defs import Path, CanonicalGrammar, ParseTree

T = TypeVar("T")
S = TypeVar("S")
Tree = Tuple[T, Optional[List['Tree[T]']]]
DictTree = Dict[T, 'DictTree']


def dict_tree_from_paths(paths: Sequence[Sequence[T]]) -> DictTree[S]:
    def add_to_tree(root: DictTree, branch: Sequence[T]):
        if not branch:
            return
        if branch[0] not in root:
            root[branch[0]] = {}
        add_to_tree(root[branch[0]], branch[1:])

    tree: DictTree = {}
    for path in paths:
        add_to_tree(tree, path)

    assert len(tree) == 1  # Rooted tree

    return tree


def tree_from_paths(paths: Sequence[Sequence[T]]) -> Tree[S]:
    return dict_tree_to_tree(dict_tree_from_paths(paths))


def dict_tree_to_tree(tree: DictTree[T]) -> Tree[T]:
    assert len(tree) == 1  # Rooted tree
    label: T = next(iter(tree))
    return label, [dict_tree_to_tree(child) for child in [{t[0]: t[1]} for t in tree[label].items()]]


def dfs(tree: ParseTree, action: Callable[[ParseTree], None] = print):
    node, children = tree
    action(tree)
    if children is not None:
        for child in children:
            dfs(child, action)


def get_subtree(tree: Tree[T], path: Path) -> Tree[T]:
    """Access a subtree based on `path` (a list of children numbers)"""
    curr_node = tree
    while path:
        curr_node = curr_node[1][path[0]]
        path = path[1:]

    return curr_node


def replace_path(
        tree: Tree[T],
        path: Path,
        replacement_tree: Tree[T]) -> Tree[T]:
    """Returns tree where replacement_tree has been inserted at `path` instead of the original subtree"""
    stack: List[Tree[T]] = [tree]
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
        tree: Tree,
        f: Callable[[Tree], bool],
        enforce_unique: bool = False) -> List[Tuple[Path, Tree]]:
    result: List[Tuple[Path, Tree]] = []

    for path, subtree in tree_paths(tree):
        if f(subtree):
            result.append((path, subtree))

            if enforce_unique and len(result) > 1:
                raise RuntimeError(f"Found searched-for element more than once in {tree}")

    return result


TRAVERSE_PREORDER = 0
TRAVERSE_POSTORDER = 1


def traverse_tree(
        tree: Tree[T],
        action: Callable[[Path, Tree[T]], None],
        abort_condition: Callable[[Path, Tree[T]], bool] = lambda p, n: False,
        kind: int = TRAVERSE_PREORDER,
        reverse: bool = False) -> None:
    stack_1: List[Tuple[Path, Tree[T]]] = [((), tree)]
    stack_2: List[Tuple[Path, Tree[T]]] = []

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


def tree_paths(tree: Tree[T]) -> List[Tuple[Path, Tree[T]]]:
    def action(path, node):
        result.append((path, node))

    result: List[Tuple[Path, Tree[T]]] = []
    traverse_tree(tree, action, kind=TRAVERSE_PREORDER)
    return result


def trie_from_parse_tree(tree: Tree) -> datrie.Trie:
    trie = mk_subtree_trie()  # Works for up to 30 children of a node
    for path, subtree in tree_paths(tree):
        trie[path_to_trie_key(path)] = (path, subtree)
    return trie


def next_trie_key(trie: datrie.Trie, path: Path | str) -> Optional[str]:
    if isinstance(path, tuple):
        path = path_to_trie_key(path)

    suffixes = tuple(filter(None, trie.suffixes(path)))
    if suffixes:
        return path + suffixes[0]
    else:
        prefixes = reversed(trie.prefixes(path))
        for prefix in prefixes:
            maybe_next_key = prefix[:-1] + chr(ord(prefix[-1]) + 1)
            if trie.keys(maybe_next_key):
                return maybe_next_key

        return None


def mk_subtree_trie() -> datrie.Trie:
    return datrie.Trie([chr(i) for i in range(30)])


def get_subtrie(trie: datrie.Trie, new_root_path: Path | str) -> datrie.Trie:
    subtrees_trie = mk_subtree_trie()

    if isinstance(new_root_path, str):
        root_key = new_root_path
        root_path_len = len(root_key) - 1
    else:
        assert isinstance(new_root_path, tuple)
        root_key = path_to_trie_key(new_root_path)
        root_path_len = len(new_root_path)

    for suffix in trie.suffixes(root_key):
        path, tree = trie[root_key + suffix]
        subtrees_trie[chr(1) + suffix] = (path[root_path_len:], tree)

    return subtrees_trie


def tree_to_string(tree: Tree, show_open_leaves: bool = False) -> str:
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


def open_leaves(tree: Tree) -> Generator[Tuple[Path, Tree], None, None]:
    return ((path, sub_tree) for path, sub_tree in tree_paths(tree) if sub_tree[1] is None)


def tree_leaves(tree: Tree) -> Generator[Tuple[Path, Tree], None, None]:
    return ((path, sub_tree) for path, sub_tree in tree_paths(tree) if not sub_tree[1])


def expand_tree(
        tree: Tree,
        canonical_grammar: CanonicalGrammar,
        limit: Optional[int] = None,
        expand_beyond_nonterminals: bool = False) -> List[Tree]:
    nonterminal_expansions = {
        (leaf_path, leaf_node): [
            [(child, None if is_nonterminal(child) else [])
             for child in expansion]
            for expansion in canonical_grammar[leaf_node[0]]
            if expand_beyond_nonterminals or any(is_nonterminal(child) for child in expansion)
        ]
        for leaf_path, leaf_node in open_leaves(tree)
    }

    if all(not expansion for expansion in nonterminal_expansions.values()):
        return []

    result: List[Tree] = [tree]
    for (leaf_path, leaf_node), expansions in nonterminal_expansions.items():
        previous_result = result
        result = []
        for t in previous_result:
            if not expansions:
                result.append(t)

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

    # assert ((limit and len(result) == limit) or
    #         len(result) == math.prod(len(values) for values in nonterminal_expansions.values()))

    return result
