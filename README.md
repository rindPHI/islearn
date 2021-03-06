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

Let `prop` be a function returning `True` if a given statement evaluates normally,
and `False` if the evaluation raises an exception. The latter is the case if a
variable appears as a right-hand-side, but has not been assigned a variable before.

ISLearn can learn this property based on the "Def-Use (reST Strict)" pattern from
the catalog (the standard catalog is in `src/islearn/patterns.toml`). You call
the learner as follows:

```python
from islearn.learner import InvariantLearner
from isla.language import ISLaUnparser, Formula
from typing import Dict, Tuple

result: Dict[Formula, Tuple[float, float]] = InvariantLearner(
    LANG_GRAMMAR,
    prop=prop,
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
(1.0, 1.0): forall <assgn> assgn_1="{<var> lhs_1} := {<var> var}" in start:
  exists <assgn> assgn_2="{<var> lhs_2} := {<rhs> rhs_2}" in start:
    (before(assgn_2, assgn_1) and (= lhs_2 var))
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
| max_conjunction_size                        | 2          | Size of conjunctions to be generated. As default, conjunctions of size 1. Potentially increases specificity. |
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

## Build, Run, Install

ISLearn depends on Python 3.10 and the Python header files (from package python3.10-dev in Ubuntu Linux). Furthermore,
python3.10-venv is required to run ISLa in a virtual environment.

On Ubuntu Linux, the dependencies can be installed using

```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3.10 python3.10-dev python3.10-venv
```

For development and testing, we recommend to use ISLearn inside a virtual environment (virtualenv).
By thing the following steps in a standard shell (bash), one can run the ISLearn tests:

```shell
git clone git@github.com:rindPHI/islearn.git
cd islearn/

python3.10 -m venv venv
source venv/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install z3-solver=4.8.14.0

# Run tests
tox
```

NOTE: This projects needs z3 >= 4.8.13.0, but the requirements only list
4.8.8.0 due to a strict requirement in the fuzzingbook package. After
installing from requirements, you have to manually install a new z3 version
(e.g., `pip install z3-solver=4.8.14.0`) and simply ignore the upcoming
warning.

For running scripts without tox, you have to add the path to the ISLearn folder to the PYTHONPATH environment variable.
For running the evaluation, you also have to add the `tests/` directory.
This is done by typing (in bash)

```shell
export PYTHONPATH=$PYTHONPATH:`pwd`
export PYTHONPATH=$PYTHONPATH:`pwd`/tests
```

inside the ISLearn top-level directory. Then you can run, for example, `python3 -O evaluations/icmp/evaluate_icmp.py`.
For using ISLa in Visual Studio, you might have to set the value of the environment variable in the launch.json file; in
Pycharm, we did not have to apply any special settings.

---

To install ISLearn globally (not recommended, less well tested), run

```shell
python3 -m build
pip3 install dist/islaern-0.1a1-py3-none-any.whl
```