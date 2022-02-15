import copy
import json
import logging
import unittest

from fuzzingbook.Grammars import JSON_GRAMMAR
from fuzzingbook.Parser import EarleyParser
from isla.fuzzer import GrammarCoverageFuzzer
from isla.language import DerivationTree

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


if __name__ == '__main__':
    unittest.main()
