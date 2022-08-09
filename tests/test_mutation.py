import json
import logging
import math
import string
import unittest

from isla.evaluator import evaluate
from isla.helpers import srange
from isla.language import DerivationTree
from isla.parser import EarleyParser
from isla_formalizations import scriptsizec

from islearn.mutation import MutationFuzzer
from islearn_example_languages import JSON_GRAMMAR


class TestMutator(unittest.TestCase):
    logger = logging.getLogger("TestMutator")

    def test_json(self):
        def prop(tree: DerivationTree) -> bool:
            json_obj = json.loads(str(tree))
            return isinstance(json_obj, dict) and "key" in json_obj

        inputs = [' { "key" : 13 } ']
        trees = [DerivationTree.from_parse_tree(next(EarleyParser(JSON_GRAMMAR).parse(inp)))
                 for inp in inputs]

        mutation_fuzzer = MutationFuzzer(JSON_GRAMMAR, trees, prop, k=4)
        # mutation_fuzzer = MutationFuzzer(grammar, trees, lambda t: not prop(t), k=4, max_mutations=2)
        for inp in mutation_fuzzer.run(extend_fragments=False):
            self.assertTrue(prop(inp))
            TestMutator.logger.info(str(inp))

    def test_mutate_scriptsize_c(self):
        correct_property = """
        forall <expr> use_ctx in start:
          forall <id> use in use_ctx:
            exists <declaration> def_ctx in start:
              exists <id> def in def_ctx:
                (before(def_ctx, use_ctx) and
                (= def use))"""

        def prop(tree: DerivationTree) -> bool:
            return evaluate(correct_property, tree, scriptsizec.SCRIPTSIZE_C_GRAMMAR).is_true()

        # def prop(tree: DerivationTree) -> bool:
        #     return compile_scriptsizec_clang(tree) is True

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        mutation_fuzzer = MutationFuzzer(scriptsizec.SCRIPTSIZE_C_GRAMMAR, inputs, prop, k=3)
        for inp in mutation_fuzzer.run(num_iterations=50, alpha=.1):
            self.assertTrue(prop(inp))
            logging.getLogger(type(self).__name__).info(inp)

    def test_mutate_arith_grammar(self):
        grammar = {
            "<start>": ["<arith_expr>"],
            "<arith_expr>": ["<function>(<number>)"],
            "<function>": ["sqrt", "sin", "cos", "tan"],
            "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
            "<maybe_minus>": ["", "-"],
            "<onenine>": [str(num) for num in range(1, 10)],
            "<digit>": srange(string.digits),
            "<maybe_digits>": ["", "<digits>"],
            "<digits>": ["<digit>", "<digit><digits>"],
            "<maybe_frac>": ["", ".<digits>"]
        }

        def arith_eval(inp: DerivationTree) -> float:
            return eval(str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan})

        def prop(inp: DerivationTree) -> bool:
            try:
                arith_eval(inp)
                return False
            except ValueError:
                return True

        raw_inputs = ["cos(-2)"]
        inputs = [
            DerivationTree.from_parse_tree(
                next(EarleyParser(grammar).parse(inp)))
            for inp in raw_inputs]

        mutation_fuzzer = MutationFuzzer(grammar, inputs, prop, k=3)
        for inp in mutation_fuzzer.run(num_iterations=100, alpha=.1):
            self.assertTrue(prop(inp))
            logging.getLogger(type(self).__name__).info(inp)


if __name__ == '__main__':
    unittest.main()
