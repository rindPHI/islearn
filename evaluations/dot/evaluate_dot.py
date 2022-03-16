import logging
import os
import dill as pickle
import re
import urllib.request

from fuzzingbook.Parser import PEGParser
from isla import language
from isla.language import ISLaUnparser

from islearn.learner import InvariantLearner
from islearn.reducer import InputReducer
from languages import DOT_GRAMMAR, render_dot

logging.basicConfig(level=logging.DEBUG)


def prop(tree: language.DerivationTree) -> bool:
    return render_dot(tree) is True


dirname = os.path.abspath(os.path.dirname(__file__))
parser = PEGParser(DOT_GRAMMAR)
reducer = InputReducer(DOT_GRAMMAR, prop, k=3)

# TODO: Obtain some more DOT files
urls = [
    "https://raw.githubusercontent.com/ecliptik/qmk_firmware-germ/56ea98a6e5451e102d943a539a6920eb9cba1919/users/dennytom/chording_engine/state_machine.dot",
    "https://raw.githubusercontent.com/Ranjith32/linux-socfpga/30f69d2abfa285ad9138d24d55b82bf4838f56c7/Documentation/blockdev/drbd/disk-states-8.dot",
    # Below one is graph, not digraph
    "https://raw.githubusercontent.com/nathanaelle/wireguard-topology/f0e42d240624ca0aa801d890c1a4d03d5901dbab/examples/3-networks/topology.dot"
]

positive_trees = []
reduced_trees = []

for url in urls:
    file_name = url.split("/")[-1]
    tree_file = f"{dirname}/inputs/{file_name}.tree"
    reduced_tree_file = f"{dirname}/inputs/{file_name}.reduced.tree"

    if os.path.isfile(tree_file) and os.path.isfile(reduced_tree_file):
        with open(tree_file, 'rb') as file:
            positive_trees.append(pickle.loads(file.read()))

        with open(reduced_tree_file, 'rb') as file:
            reduced_trees.append(pickle.loads(file.read()))

        continue

    with urllib.request.urlopen(url) as f:
        # The XML grammar is a little simplified, so we remove some elements.
        dot_code: str = f.read().decode('utf-8').strip()
        dot_code = re.sub(r"(^|\n)\s*//.*?(\n|$)", "", dot_code)
        dot_code = dot_code.replace("\\n", "\n")
        dot_code = dot_code.replace("\r\n", "\n")

        # Make sure we still have valid DOT.
        assert render_dot(dot_code) is True

    sample_tree = language.DerivationTree.from_parse_tree(list(parser.parse(dot_code))[0])
    reduced_tree = reducer.reduce_by_smallest_subtree_replacement(sample_tree)

    with open(tree_file, 'wb') as file:
        file.write(pickle.dumps(sample_tree))

    with open(reduced_tree_file, 'wb') as file:
        file.write(pickle.dumps(reduced_tree))

    positive_trees.append(sample_tree)
    reduced_trees.append(reduced_tree)

assert len(positive_trees) == len(urls)
assert len(reduced_trees) == len(urls)

# Learn invariants
result = InvariantLearner(
    DOT_GRAMMAR,
    prop=prop,
    activated_patterns={"String Existence"},
    positive_examples=reduced_trees,
    target_number_positive_samples=15,
    target_number_negative_samples=20,
    max_disjunction_size=2,
    filter_inputs_for_learning_by_kpaths=False,
    min_recall=1,
    min_precision=.8,
    reduce_inputs_for_learning=False,
    generate_new_learning_samples=False,
    exclude_nonterminals={
        "<WS>", "<WSS>", "<MWSS>",
        "<esc_or_no_string_endings>", "<esc_or_no_string_ending>", "<no_string_ending>", "<LETTER_OR_DIGITS>",
        "<LETTER>", "<maybe_minus>", "<maybe_comma>", "<maybe_semi>"
    }
).learn_invariants(ensure_unique_var_names=False)

# print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

best_invariant, (precision, recall) = next(iter(result.items()))
print(f"Best invariant (*estimated* precision {precision:.2f}, recall {recall:.2f}):")
print(ISLaUnparser(best_invariant).unparse())
