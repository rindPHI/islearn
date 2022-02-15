import copy
import json
import logging
import unittest

from fuzzingbook.Grammars import JSON_GRAMMAR
from fuzzingbook.Parser import EarleyParser
from isla.evaluator import evaluate
from isla.fuzzer import GrammarCoverageFuzzer
from isla.language import DerivationTree
from isla_formalizations import scriptsizec
from isla_formalizations.scriptsizec import compile_scriptsizec_clang

from islearn.mutation import MutationFuzzer


class TestMutator(unittest.TestCase):
    logger = logging.getLogger("TestMutator")

    def test_json(self):
        def prop(tree: DerivationTree) -> bool:
            json_obj = json.loads(str(tree))
            return isinstance(json_obj, dict) and "key" in json_obj

        grammar = copy.deepcopy(JSON_GRAMMAR)
        grammar["<value>"] = ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"]
        grammar["<int>"] = ["<digit>", "<onenine><digits>", "-<digit>", "-<onenine><digits>"]

        inputs = [' { "key" : 13 } ']
        trees = [DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
                 for inp in inputs]

        mutation_fuzzer = MutationFuzzer(grammar, trees, prop, k=4)
        # mutation_fuzzer = MutationFuzzer(grammar, trees, lambda t: not prop(t), k=4, max_mutations=2)
        for inp in mutation_fuzzer.run(extend_fragments=False):
            self.assertTrue(prop(inp))
            TestMutator.logger.info(str(inp))

    def test_mutate_scriptsize_c(self):
        def prop(tree: DerivationTree) -> bool:
            return compile_scriptsizec_clang(tree) is True

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


if __name__ == '__main__':
    unittest.main()
