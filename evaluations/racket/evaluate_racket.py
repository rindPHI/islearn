import logging
import os
import re
import sys
import urllib.request

import dill as pickle
import isla.fuzzer
from fuzzingbook.Parser import PEGParser
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.language import DerivationTree, ISLaUnparser, parse_isla

from islearn.helpers import tree_in
from islearn.language import parse_abstract_isla
from islearn.learner import InvariantLearner
from islearn.mutation import MutationFuzzer
from languages import RACKET_BSL_GRAMMAR, load_racket

logging.basicConfig(level=logging.DEBUG)


def prop(tree: language.DerivationTree) -> bool:
    return load_racket(tree) is True


dirname = os.path.abspath(os.path.dirname(__file__))
parser = PEGParser(RACKET_BSL_GRAMMAR)
graph = gg.GrammarGraph.from_grammar(RACKET_BSL_GRAMMAR)

# The racket syntax check is really expensive; therefore, reduction cannot be used efficiently.
# Also, the HTDP examples are pretty small already.
urls = [
    f"https://github.com/johnamata/compsci/raw/"
    f"cfb0e48c151da1d3463f3f0faca9f666af22ee16/htdp/exercises/{str(i).rjust(3, '0')}.rkt"
    for i in range(1, 30)
]

positive_trees = []

for url in urls:
    file_name = url.split("/")[-1]
    tree_file = f"{dirname}/inputs/{file_name}.tree"

    if os.path.isfile(tree_file):
        with open(tree_file, 'rb') as file:
            positive_trees.append(pickle.loads(file.read()))

        continue

    with urllib.request.urlopen(url) as f:
        racket_code = f.read().decode('utf-8')
        racket_code = racket_code.replace("\\n", "\n")
        racket_code = racket_code.replace("\r\n", "\n").strip()
        # We remove comments, since they sometimes lead to syntactically invalid inputs produced
        # by the mutation fuzzer, since parts of an expression occur after the comment.
        racket_code = re.sub(r';.*(\n|$)', '', racket_code).strip()

        if "GRacket" in racket_code:  # Not a real racket file
            continue

        # Make sure we still have valid Racket.
        assert load_racket(racket_code) is True, f"URL {url} is invalid, code:\n{racket_code}"

    try:
        sample_tree = language.DerivationTree.from_parse_tree(list(parser.parse(racket_code))[0])
    except SyntaxError:
        print(f"URL {url} is invalid, code:\n{racket_code}")
        sys.exit(1)

    with open(tree_file, 'wb') as file:
        file.write(pickle.dumps(sample_tree))

    positive_trees.append(sample_tree)

# Generate inputs for validation, and further (negative) training inputs.
# This is to exclude certain negative inputs that are actually manifestations
# of grammar imprecision rather than semantic errors. The learner's fuzzer
# cannot tell those apart.
target_number_positive_inputs = 50
target_number_negative_inputs = 70

new_positive_trees = []
negative_trees = []

# We run two mutation fuzzers and a grammar fuzzer in parallel
mutation_fuzzer: MutationFuzzer = MutationFuzzer(RACKET_BSL_GRAMMAR, positive_trees, prop, k=3)
mutate_fuzz = mutation_fuzzer.run(None, alpha=.1, yield_negative=True)

grammar_fuzzer = isla.fuzzer.GrammarCoverageFuzzer(RACKET_BSL_GRAMMAR)

# These are *syntactic* errors that we do not consider; they result from grammar imprecisions.
syntactic_errors = [
    'expected a closing `"`',
    'expected a `)` to close `(`',
    'unexpected `)`',
    'bad syntax',
    'illegal use of `',
    'not a proper list',
    'bad module-path string',
    'not a require sub-form',
    'missing procedure expression',
    'expected a `)` to close `(`',
    'expected only alphanumeric',
    'end-of-file following',
    '`#lang` not enabled',
    'expected a function call, but there is no open parenthesis before this function',
    "expected a name that does not end",
    "expected a string for a lib path, found something else",
    "expected a function after the open parenthesis",
    "expected a field-specification keyword",
    "not an identifier",
    "missing `]` to close `[`",
    "expected `)` to close preceding `(`",
    "a module-naming string cannot be empty",
    "expected a `]` to close `[`",
    "non-pair found in list",
    "name is not provided",
]

i = 0
while (len(new_positive_trees) + len(positive_trees) // 2 < target_number_positive_inputs
       or len(negative_trees) < target_number_negative_inputs):
    if i % 10 == 0:
        print(f"Fuzzing: {len(new_positive_trees):02} positive / {len(negative_trees):02} negative inputs")

    fuzzer_inputs = [
        next(mutate_fuzz),
        grammar_fuzzer.expand_tree(DerivationTree("<start>", None))
    ]

    for idx, inp in enumerate(fuzzer_inputs):
        eval_result = load_racket(inp)
        if (isinstance(eval_result, str) and
                any(syntactic_error in eval_result for syntactic_error in syntactic_errors)):
            continue

        # if isinstance(eval_result, str):
        #     print("===== REPORTED ERROR =====")
        #     print("Program:")
        #     print(inp)
        #     print()
        #     print("Message:")
        #     print(eval_result)
        #     print("=================")

        if (len(new_positive_trees) + len(positive_trees) // 2 < target_number_positive_inputs and
                eval_result is True and
                not tree_in(inp, new_positive_trees)):
            new_positive_trees.append(inp)
            if idx == 0:
                mutation_fuzzer.population.add(inp)
        elif (len(negative_trees) < target_number_negative_inputs and
              not eval_result is True and
              not tree_in(inp, negative_trees)):
            negative_trees.append(inp)

    i += 1

learning_inputs = positive_trees[:len(positive_trees) // 2]
negative_learning_inputs = negative_trees[:20]

positive_validation_inputs = positive_trees[len(positive_trees) // 2:] + new_positive_trees
negative_validation_inputs = negative_trees[20:]

# for inp in learning_inputs:
#     print("INPUT")
#     print("=====\n")
#     print(inp)

# Learn invariants
# result = InvariantLearner(
#     RACKET_BSL_GRAMMAR,
#     prop=prop,
#     activated_patterns={
#         "Def-Use (XML-Attr Disjunctive)",
#         "Def-Use (reST Strict Reserved Names)",  # Deactivate this to evaluate w/o the custom extension to the catalog
#     },
#     positive_examples=learning_inputs,
#     negative_examples=negative_learning_inputs,
#     target_number_positive_samples=15,
#     target_number_negative_samples=20,
#     max_conjunction_size=2,
#     max_disjunction_size=1,
#     filter_inputs_for_learning_by_kpaths=False,
#     min_recall=.8,
#     min_precision=.8,
#     reduce_inputs_for_learning=False,
#     do_generate_more_inputs=False,
#     mexpr_expansion_limit=1,
#     max_nonterminals_in_mexpr=9,
#     exclude_nonterminals={
#         "<maybe_wss_names>",
#         "<wss_exprs>",
#         "<maybe_cond_args>",
#         "<strings_mwss>",
#         "<NAME_CHAR>",
#         "<ONENINE>",
#         "<ESC_OR_NO_STRING_ENDINGS>",
#         "<ESC_OR_NO_STRING_ENDING>",
#         "<NO_STRING_ENDING>",
#         "<CHARACTER>",
#         "<DIGIT>",
#         "<LETTERORDIGIT>",
#         "<MWSS>",
#         "<WSS>",
#         "<WS>",
#         "<maybe_comments>",
#         "<COMMENT>",
#         "<HASHDIRECTIVE>",
#         "<NOBR>",
#         "<NOBRs>",
#         "<test_case>",
#         "<library_require>",
#         "<pkg>",
#         "<SYMBOL>",
#         "<NUMBER>",
#         "<DIGITS>",
#         "<MAYBE_DIGITS>",
#         "<INT>",
#         "<BOOLEAN>",
#         "<STRING>",
#         "<program>",  # TODO: Remove for evaluation
#         "<def_or_exprs>",  # TODO: Remove for evaluation
#         "<def_or_expr>",  # TODO: Remove for evaluation
#         "<cond_args>",  # TODO: Remove for evaluation
#     }
# ).learn_invariants()
#
# print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
#
# best_invariant, (specificity, sensitivity) = next(iter(result.items()))
# print(f"Best invariant (*estimated* specificity {specificity:.2f}, sensitivity {sensitivity:.2f}):")
# print(ISLaUnparser(best_invariant).unparse())

best_invariant = parse_abstract_isla("""
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use "sqrt") or
   (= prefix_use "string-append") or
   (= prefix_use "substring") or
   (= prefix_use "add-line") or
   (= prefix_use "right-triangle") or
   (= prefix_use "ellipse") or
   (= prefix_use "rectangle") or
   (= prefix_use "overlay/align/offset") or
   (= prefix_use "overlay") or
   (= prefix_use "or") or
   (= prefix_use "not") or
   (= prefix_use "cond") or
   (= prefix_use "string?") or
   (= prefix_use "number?") or
   (= prefix_use "boolean?") or
   (= prefix_use "image?") or
   (= prefix_use "*") or
   (= prefix_use "<=") or
   (= prefix_use "-") or
   (= prefix_use "string-length") or
   (= prefix_use "+") or
   (= prefix_use "string-ith") or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS><expr><MWSS>)" in start:
    (inside(attribute, outer_tag) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))
""", RACKET_BSL_GRAMMAR)


# NOTE: The following semantic errors are not covered by our invariant:
#       - "this function is not defined," e.g., for `(re2uire 2htdp/image)`
#       - "collection not found," e.g., for `(require U)`
#       - "cannot open module file,", e.g., for `(require 2htdp/imaQ)`
#       - "couldn't find teachpacks in table,", e.g., for `#reader(lib "htdp-beginner-reader.ss" "lang")((XYZ))`
#       - "this name was defined in the language or a required library and cannot be re-defined," e.g.,
#         for `(define - ...)`
#
# This impacts the specificity results.

# Finally, obtain confusion matrix entries.
tp, tn, fp, fn = 0, 0, 0, 0

for inp in positive_validation_inputs:
    if evaluate(best_invariant, inp, RACKET_BSL_GRAMMAR, graph=graph).is_true():
        tp += 1
    else:
        fn += 1

for inp in negative_validation_inputs:
    if evaluate(best_invariant, inp, RACKET_BSL_GRAMMAR, graph=graph).is_true():
        fp += 1
    else:
        tn += 1

print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
