import copy
import json
import unittest

from fuzzingbook.Grammars import JSON_GRAMMAR
from fuzzingbook.Parser import EarleyParser
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.isla_predicates import STANDARD_SEMANTIC_PREDICATES
from isla.language import parse_isla, ISLaUnparser
from isla_formalizations import scriptsizec, csv

from islearn.learner import generate_candidates, patterns_from_file, learn_invariants, chain_implies


class TestLearner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_grammar = copy.deepcopy(JSON_GRAMMAR)
        self.json_grammar["<value>"] = ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"]
        self.json_grammar["<int>"] = ["<digit>", "<onenine><digits>", "-<digit>", "-<onenine><digits>"]

    def test_learn_invariants_simple_scriptsize_c(self):
        correct_property = """
forall <expr> use_ctx in start:
  forall <id> use in use_ctx:
    exists <declaration> def_ctx in start:
      exists <id> def in def_ctx:
        (before(def_ctx, use_ctx) and
        (= def use))"""

        def prop(tree: language.DerivationTree) -> bool:
            return scriptsizec.compile_scriptsizec_clang(tree) is True

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        result = learn_invariants(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            prop,
            activated_patterns={"Def-Use 1"},
            positive_examples=inputs
        )

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    def test_filter_invariants_simple_csv_colno(self):
        pattern = """
exists int num:
  forall <?NONTERMINAL> elem in start:
    ((>= (str.to.int num) 1) and
    count(elem, <?NONTERMINAL>, num))"""

        correct_property = """
exists int num:
  forall <csv-record> elem in start:
    ((>= (str.to.int num) 1) and
    count(elem, "<raw-field>", num))"""

        raw_inputs = [
            """1a;\"2   a\";\" 12\"
4; 55;6
123;1;1
""",
            """  1;2 
""",
            """1;3;17
12;" 123";"  2"
""",
        ]

        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(csv.CSV_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        candidates = generate_candidates([pattern], inputs, csv.CSV_GRAMMAR)
        # print("\n".join(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

        self.assertIn(correct_property.strip(), list(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

    def test_learn_invariants_simple_csv_colno(self):
        correct_property = """
exists int num:
  forall <csv-record> elem in start:
    ((>= (str.to.int num) 1) and
    count(elem, "<raw-field>", num))"""

        def prop(tree: language.DerivationTree) -> bool:
            return evaluate(correct_property, tree, csv.CSV_GRAMMAR).is_true()

        result = learn_invariants(
            csv.CSV_GRAMMAR,
            prop,
            activated_patterns={"Equal Count"},
        )

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            parse_isla(correct_property, csv.CSV_GRAMMAR, semantic_predicates=STANDARD_SEMANTIC_PREDICATES),
            result)

        perfect_precision_formulas = [f for f, p in result.items() if p == 1]
        self.assertEqual(2, len(perfect_precision_formulas))
        self.assertIn(correct_property.strip(), [ISLaUnparser(f).unparse() for f in perfect_precision_formulas])

    def test_string_existence(self):
        correct_property = """
forall <json> container in start:
  exists <string> elem in container:
    (= elem \"""key""\")
"""

        def prop(tree: language.DerivationTree) -> bool:
            json_obj = json.loads(str(tree))
            return isinstance(json_obj, dict) and "key" in json_obj

        inputs = [
            ' { "key" : 13 } ',
            # ' { "asdf" : [ 26 ] , "key" : "x" } ',
        ]
        trees = [language.DerivationTree.from_parse_tree(next(EarleyParser(self.json_grammar).parse(inp)))
                 for inp in inputs]

        # candidates = generate_candidates(
        #     [patterns_from_file()["String Existence"]],
        #     trees,
        #     self.json_grammar
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return

        self.assertTrue(all(evaluate(correct_property, tree, self.json_grammar) for tree in trees))

        result = learn_invariants(
            self.json_grammar,
            prop,
            activated_patterns={"String Existence"},
            positive_examples=trees
        )

        # print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            list(map(lambda f: ISLaUnparser(f).unparse(), [r for r, p in result.items() if p == 1.0])))

    def test_chain_implies(self):
        graph = gg.GrammarGraph.from_grammar(self.json_grammar)

        chain_1 = ("<json>", "<array>", "<digits>")
        chain_2 = ("<json>", "<object>", "<digits>")
        self.assertFalse(chain_implies(chain_1, chain_2, graph))
        self.assertFalse(chain_implies(chain_2, chain_1, graph))

        chain_1 = ("<json>", "<int>", "<digit>")
        chain_2 = ("<element>", "<int>", "<digit>")
        self.assertTrue(chain_implies(chain_1, chain_2, graph))
        self.assertFalse(chain_implies(chain_2, chain_1, graph))

        chain_1 = ("<json>", "<number>", "<digits>")
        chain_2 = ("<member>", "<members>", "<digits>")
        self.assertFalse(chain_implies(chain_1, chain_2, graph))
        self.assertFalse(chain_implies(chain_2, chain_1, graph))

        chain_1 = ("<json>", "<array>", "<string>")
        chain_2 = ("<json>", "<object>", "<string>")
        self.assertFalse(chain_implies(chain_1, chain_2, graph))
        self.assertFalse(chain_implies(chain_2, chain_1, graph))

    def test_load_patterns_from_file(self):
        patterns = patterns_from_file()
        self.assertTrue(patterns)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertIn("Def-Use", patterns)
        self.assertIn("Def-Use 1", patterns)
        self.assertIn("Def-Use 2", patterns)
        self.assertNotIn("Def-Use 9", patterns)


if __name__ == '__main__':
    unittest.main()
