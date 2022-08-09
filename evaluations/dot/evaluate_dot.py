import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

import dill as pickle
import isla.fuzzer
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.language import ISLaUnparser, DerivationTree
from isla.parser import PEGParser

from islearn.helpers import tree_in
from islearn.learner import InvariantLearner
from islearn.mutation import MutationFuzzer
from islearn.reducer import InputReducer
from islearn_example_languages import render_dot, DOT_GRAMMAR

logging.basicConfig(level=logging.DEBUG)


def prop(tree: language.DerivationTree) -> bool:
    return render_dot(tree) is True


dirname = os.path.abspath(os.path.dirname(__file__))
parser = PEGParser(DOT_GRAMMAR)
reducer = InputReducer(DOT_GRAMMAR, prop, k=3)
graph = gg.GrammarGraph.from_grammar(DOT_GRAMMAR)

Path(f"{dirname}/inputs/").mkdir(parents=False, exist_ok=True)

###
# dot_code = """
# """.strip()
# sample_tree = language.DerivationTree.from_parse_tree(list(parser.parse(dot_code))[0])
# sys.exit(0)
###


urls = [
    "https://raw.githubusercontent.com/ecliptik/qmk_firmware-germ/56ea98a6e5451e102d943a539a6920eb9cba1919/users/dennytom/chording_engine/state_machine.dot",
    "https://raw.githubusercontent.com/Ranjith32/linux-socfpga/30f69d2abfa285ad9138d24d55b82bf4838f56c7/Documentation/blockdev/drbd/disk-states-8.dot",
    "https://raw.githubusercontent.com/gmj93/hostap/d0deb2a2edf11acd6eb6440336406228eeeab96e/doc/p2p_sm.dot",
    # Below ones are graph, not digraph
    "https://raw.githubusercontent.com/nathanaelle/wireguard-topology/f0e42d240624ca0aa801d890c1a4d03d5901dbab/examples/3-networks/topology.dot",
    "https://raw.githubusercontent.com/210296kaczmarek/student-forum-poprawione/55790569976d4e92a32d9471d3549943011fcb70/vendor/bundle/ruby/2.4.0/gems/ruby-graphviz-1.2.3/examples/dot/genetic.dot",
    "https://raw.githubusercontent.com/Cloudofyou/tt-demo/5504ac17790d3863bf036f6ce8d651a862fa6b0f/tt-demo.dot",
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
        # The DOT grammar does not contain comments, so we remove them.
        dot_code: str = f.read().decode('utf-8').strip()
        dot_code = re.sub(r"(^|\n)\s*//.*?(\n|$)", "", dot_code)
        dot_code = dot_code.replace("\\n", "\n")
        dot_code = dot_code.replace("\r\n", "\n")
        dot_code = re.compile(r'/\*.*?\*/', re.DOTALL).sub("", dot_code)

        # Make sure we still have valid DOT.
        assert render_dot(dot_code) is True, f"URL {url} is invalid, code:\n{dot_code}"

    try:
        sample_tree = language.DerivationTree.from_parse_tree(list(parser.parse(dot_code))[0])
    except SyntaxError:
        print(f"URL {url} is invalid, code:\n{dot_code}")
        sys.exit(1)

    reduced_tree = reducer.reduce_by_smallest_subtree_replacement(sample_tree)

    with open(tree_file, 'wb') as file:
        file.write(pickle.dumps(sample_tree))

    with open(reduced_tree_file, 'wb') as file:
        file.write(pickle.dumps(reduced_tree))

    positive_trees.append(sample_tree)
    reduced_trees.append(reduced_tree)

assert len(positive_trees) == len(urls)
assert len(reduced_trees) == len(urls)

learning_inputs = reduced_trees[:3]
validation_inputs = positive_trees[3:]

# Learn invariants
result = InvariantLearner(
    DOT_GRAMMAR,
    prop=prop,
    activated_patterns={"String Existence"},
    positive_examples=reduced_trees,
    target_number_positive_samples=50,
    target_number_negative_samples=50,
    max_disjunction_size=2,
    filter_inputs_for_learning_by_kpaths=False,
    min_recall=1,
    min_specificity=.8,
    reduce_inputs_for_learning=False,
    generate_new_learning_samples=False,
    exclude_nonterminals={
        "<WS>", "<WSS>", "<MWSS>",
        "<esc_or_no_string_endings>", "<esc_or_no_string_ending>", "<no_string_ending>", "<LETTER_OR_DIGITS>",
        "<LETTER>", "<maybe_minus>", "<maybe_comma>", "<maybe_semi>"
    }
).learn_invariants()

print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

best_invariant, (specificity, sensitivity) = next(iter(result.items()))
print(f"Best invariant (*estimated* specificity {specificity:.2f}, sensitivity: {sensitivity:.2f}):")
print(ISLaUnparser(best_invariant).unparse())

# Generate inputs for validation
target_number_positive_inputs = 50
target_number_negative_inputs = 50

negative_validation_inputs = []

# We run two mutation fuzzers and a grammar fuzzer in parallel
mutation_fuzzer = MutationFuzzer(DOT_GRAMMAR, learning_inputs, prop, k=3)
mutate_fuzz = mutation_fuzzer.run(500, alpha=.1, yield_negative=True)

grammar_fuzzer = isla.fuzzer.GrammarCoverageFuzzer(DOT_GRAMMAR)

i = 0
while (len(validation_inputs) < target_number_positive_inputs
       or len(negative_validation_inputs) < target_number_negative_inputs):
    if i % 10 == 0:
        print(f"Fuzzing: {len(validation_inputs):02} positive / {len(negative_validation_inputs):02} negative inputs")

    fuzzer_inputs = [
        next(mutate_fuzz),
        grammar_fuzzer.expand_tree(DerivationTree("<start>", None))
    ]

    for idx, inp in enumerate(fuzzer_inputs):
        if (len(validation_inputs) < target_number_positive_inputs and
                prop(inp) and
                not tree_in(inp, validation_inputs)):
            validation_inputs.append(inp)
            if idx == 0:
                mutation_fuzzer.population.add(inp)
        elif (len(negative_validation_inputs) < target_number_negative_inputs and
              not prop(inp) and
              not tree_in(inp, negative_validation_inputs)):
            negative_validation_inputs.append(inp)

    i += 1

# Finally, obtain confusion matrix entries.
tp, tn, fp, fn = 0, 0, 0, 0

for inp in validation_inputs:
    if evaluate(best_invariant, inp, DOT_GRAMMAR, graph=graph).is_true():
        tp += 1
    else:
        fn += 1

for inp in negative_validation_inputs:
    if evaluate(best_invariant, inp, DOT_GRAMMAR, graph=graph).is_true():
        fp += 1
    else:
        tn += 1

print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
