# ISLearn

ISLearn is a system for learning ISLa constraints from a set of patterns.
Patterns are instantiated according to sample inputs (that can also automatically
be generated based on a given program property), filtered, and combined to
a disjunctive normal form (conjunctions of disjunctions of literals, where
literals are possibly negated, instantiated patterns). ISLearn output the result
ranked by estimates for *specificity* and *recall* of the results: Top-ranked
invariants were found to be best suitable for distinguishing positive (with
respect to a program property) and negative inputs.

## Example

Consider a grammar of a simple assignment programming language (e.g., "x := 1 ; y := x"):

```python
import string

LANG_GRAMMAR = {
    "<start>":
        ["<stmt>"],
    "<stmt>":
        ["<assgn>", "<assgn> ; <stmt>"],
    "<assgn>":
        ["<var> := <rhs>"],
    "<rhs>":
        ["<var>", "<digit>"],
    "<var>": list(string.ascii_lowercase),
    "<digit>": list(string.digits)
}
```

For learning input invariants, we need a property of the inputs satisfying that
invariant. If the goal is to learn properties about (non-context-free) syntactical
correctness of a programming language, this is a function returning `True` if, and
only if, a statement is executed without raising a specific class of errors. For
our language described by `LANG_GRAMMAR`, we define an `eval_lang` function that raises
an error if an identifier is not defined; `validate_lang` turns this into a property.

```python
from typing import Dict
from islearn.parse_tree_utils import dfs, get_subtree, tree_to_string

from isla.language import DerivationTree
from isla.parser import EarleyParser
from isla.type_defs import ParseTree

def eval_lang(inp: str) -> Dict[str, int]:
    def assgnlhs(assgn: ParseTree):
        return tree_to_string(get_subtree(assgn, (0,)))

    def assgnrhs(assgn: ParseTree):
        return tree_to_string(get_subtree(assgn, (2,)))

    valueMap: Dict[str, int] = {}
    tree = list(EarleyParser(LANG_GRAMMAR).parse(inp))[0]

    def evalAssignments(tree):
        node, children = tree
        if node == "<assgn>":
            lhs = assgnlhs(tree)
            rhs = assgnrhs(tree)
            if rhs.isdigit():
                valueMap[lhs] = int(rhs)
            else:
                valueMap[lhs] = valueMap[rhs]

    dfs(tree, evalAssignments)

    return valueMap

def validate_lang(inp: DerivationTree) -> bool:
    try:
        eval_lang(str(inp))
        return True
    except Exception:
        return False
```

ISLearn can learn this property based on the "Def-Use (reST Strict)" pattern from
the catalog (the standard catalog is in `src/islearn/patterns.toml`). You call
the learner as follows:

```python
from islearn.learner import InvariantLearner
from isla.language import ISLaUnparser, Formula
from typing import Dict, Tuple

result: Dict[Formula, Tuple[float, float]] = InvariantLearner(
    LANG_GRAMMAR,
    prop=validate_lang,
    activated_patterns={
        "Def-Use (reST Strict)",  # Optional; leads to quicker results
    },
).learn_invariants()

print("\n".join(map(
    lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
    {f: p for f, p in result.items() if p[0] > .0}.items())))
```

The expected result is

```
(1.0, 1.0): forall <rhs> use_ctx="{<var> use}" in start:
  exists <assgn> def_ctx="{<var> def} := <rhs>" in start:
    (before(def_ctx, use_ctx) and
    (= use def))
```

That invariant has full specificity (first value in the tuple) and recall
(second value), since it holds *exactly* for all valid inputs.

## Resources / Important Files

* You find the pattern catalog in `src/islearn/patterns.toml`.
* The evaluation scripts are in the directory `evaluations/`.
* The most important files of our implementation are `src/islearn/learner.py`,
  containing the learning incl. candidate generation, filtering, etc., and
  and `src/islearn/language.py`, which defines the abstract ISLa langauge for
  the patterns catalog.

## Configuration Parameters

ISLearn (the class `InvariantLearner`) has a number of optional configuration parameters that can be passed to the
constructor; many of them can be used to reduce or expand the search space, and thus either lead to more / longer
invariants or to quicker results.

| Parameter                                   | Default    | Description                                                                                                  |
|:--------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------|
| prop                                        | None       | The program property for distinguishing valid from invalid inputs.                                           |
| positive_examples                           | None       | A set of valid sample inputs. Optional if prop is present.                                                   |
| negative_examples                           | None       | A set of invalid sample inputs. Optional. Needed to estimate specificifity if prop is not present.           |
| min_recall                                  | .9         | What is the minimum target recall value?                                                                     |
| min_specificity                             | .6         | What is the minimum target specificity value?                                                                |
| max_disjunction_size                        | 1          | Size of disjunctions to be generated. As default, no disjunctions (size 1). Potentially increases recall.    |
| include_negations_in_disjunctions           | False      | Also include negations in literals within disjunctions. Use with case; can lead to "spurious" invariants.    |
| max_conjunction_size                        | 2          | Size of conjunctions to be generated. As default, conjunctions of size 2. Potentially increases specificity. |
| activated_patterns                          | None       | A set of pattern names from the catalog that should be selected. As a default, all patterns are selected.    |
| deactivated_patterns                        | None       | A set of pattern names from the catalog that should *not* be selected.                                       |
| pattern_file                                | None       | A file name to a pattern catalog (TOML file). Standard is `src/islearn/patterns.toml`.                       |
| patterns                                    | None       | A set of patterns (abstract formulas) to be considered if no catalog should be used.                         |
| target_number_positive_samples              | 10         | How many positive samples should be created using fuzzing?                                                   |
| target_number_negative_samples              | 10         | How many negative samples should be created using fuzzing?                                                   |
| target_number_positive_samples_for_learning | 10         | How many positive samples should be generated specifically for learning? <= `positive_examples`              |
| reduce_inputs_for_learning                  | True       | If `True`, learning samples are reduced, keeping k-path coverage and validity.                               |
| reduce_all_inputs                           | False      | If `True`, all inputs are reduced. Reduction of learning inputs can be specifically disabled.                |
| generate_new_learning_samples               | True       | Only if `True`, the fuzzers are used to generate further learning examples.                                  |
| do_generate_more_inputs                     | True       | Only if `True`, any further examples (positive, negative) are generated by the fuzzers.                      |
| filter_inputs_for_learning_by_kpaths        | True       | If `True`, learning inputs are filtered k-paths: Retain only those with new k-path coverage information.     |
| mexpr_expansion_limit                       | 1          | To what depth should the learner search for instantiations of match expressions?                             |
| max_nonterminals_in_mexpr                   | None       | How many nonterminals are allowed in match expressions? `None` means no restriction.                         |
| exclude_nonterminals                        | None       | "Irrelevant" grammar nonterminals. Reduces search space. Example: White space nonterminals.                  |
| perform_static_implication_check            | False      | Statically exclude weaker invariants by a translation to Z3. Rather slow, handle with care.                  |
| k                                           | 3          | The `k` from `k`-Paths. Used by the input generators and filters.                                            |

## Install, Build, Test

ISLearn depends on Python 3.10 and the Python header files. To compile all of ISLearns's dependencies, you need
gcc, g++ make, and cmake. To check out the current ISLearn version, git will be needed. Furthermore, 
python3.10-venv is required to run ISLearn in a virtual environment.

On *Alpine Linux*, all dependencies can be installed using

```shell
apk add python3.10 python3.10-dev python3.10-venv gcc g++ make cmake git 
```

### Install

To install ISLearn, a simple `pip install islearn` suffices. We recommend installing ISLearn inside a virtual
environment (virtualenv):

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install islearn
```

### Build 

ISLearn is built locally as follows:

```shell
git clone https://github.com/rindPHI/islearn.git
cd islearn/

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade build
python3 -m build
```

Then, you will find the built wheel (`*.whl`) in the `dist/` directory.

### Testing & Development

For running the ISLearn tests, you additionally need `clang`, `racket`, and `graphviz`. On Alpine Linux, you
can install those by

```shell
apk update
apk upgrade
apk add clang racket graphviz
```

Then, you can run the ISLearn tests as follows:

```shell
git clone https://github.com/rindPHI/islearn.git
cd islearn/

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Run tests
pip install -e .[test]
python3 -m pytest -n 16 tests
```