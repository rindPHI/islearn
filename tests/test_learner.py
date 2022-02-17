import copy
import json
import unittest

import pytest
from fuzzingbook.Grammars import JSON_GRAMMAR
from fuzzingbook.Parser import EarleyParser
from isla import language
from isla.evaluator import evaluate
from isla.isla_predicates import STANDARD_SEMANTIC_PREDICATES
from isla.language import parse_isla, ISLaUnparser
from isla_formalizations import scriptsizec, csv, xml_lang, rest

from islearn.learner import patterns_from_file, InvariantLearner


class TestLearner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_grammar = copy.deepcopy(JSON_GRAMMAR)
        self.json_grammar["<value>"] = ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"]
        self.json_grammar["<int>"] = ["<digit>", "<onenine><digits>", "-<digit>", "-<onenine><digits>"]

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_learn_invariants_mexpr_scriptsize_c(self):
        correct_property = """
forall <expr> use_ctx in start:
  forall <id> use in use_ctx:
    exists <declaration> def_ctx="int {<id> def};" in start:
      (before(def_ctx, use_ctx) and
      (= use def))"""

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

        #########
        # candidates = InvariantLearner(
        #     scriptsizec.SCRIPTSIZE_C_GRAMMAR,
        #     prop,
        #     activated_patterns={"Def-Use (C)"},
        #     positive_examples=inputs
        # ).generate_candidates(
        #     patterns_from_file()["Def-Use (C)"],
        #     inputs)
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##########

        result = InvariantLearner(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            prop,
            activated_patterns={"Def-Use (C)"},
            positive_examples=inputs,
            target_number_positive_samples=7,
            target_number_positive_samples_for_learning=4
        ).learn_invariants()

        # print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    def test_learn_invariants_mexpr_rest(self):
        # TODO: Inputs. Also, before is actually not true for reST!
        #       Maybe add another pattern, but then also check implications
        #       if both patterns are activated.
        correct_property = """
forall <internal_reference> use_ctx in start:
  forall <id> use in ref:
    exists <labeled_paragraph> def_ctx=".. _{<id> def}:\n\n<paragraph>" in start:
      (before(def_ctx, use_ctx) and
      (= use def))"""

        def prop(tree: language.DerivationTree) -> bool:
            return rest.render_rst(tree) is True

        raw_inputs = [
            """.. _p:

a	p_)

x
-
""",
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(rest.REST_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        ##########
        # candidates = generate_candidates(
        #     patterns_from_file()["Def-Use (reST)"],
        #     inputs,
        #     rest.REST_GRAMMAR
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##########

        result = InvariantLearner(
            rest.REST_GRAMMAR,
            prop,
            activated_patterns={"Def-Use (reST)"},
            positive_examples=inputs
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_learn_invariants_mexpr_xml(self):
        correct_property = """
forall <xml-tree> container="<{<id> opid}><inner-xml-tree></{<id> clid}>" in start:
  (= opid clid)"""

        def prop(tree: language.DerivationTree) -> bool:
            return xml_lang.validate_xml(tree) is True

        raw_inputs = [
            "<a>asdf</a>",
            "<b>xyz<c/><x>X</x></b>",
            "<a/>"
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(xml_lang.XML_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        ##########
        # candidates = generate_candidates(
        #     patterns_from_file()["Balance"],
        #     inputs,
        #     xml_lang.XML_GRAMMAR
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##########

        result = InvariantLearner(
            xml_lang.XML_GRAMMAR,
            prop,
            activated_patterns={"Balance"},
            positive_examples=inputs
        ).learn_invariants()

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

        def prop(tree: language.DerivationTree) -> bool:
            return evaluate(correct_property, tree, csv.CSV_GRAMMAR).is_true()

        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(csv.CSV_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        candidates = InvariantLearner(csv.CSV_GRAMMAR, prop, inputs).generate_candidates([pattern], inputs)
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

        result = InvariantLearner(
            csv.CSV_GRAMMAR,
            prop,
            activated_patterns={"Equal Count"},
        ).learn_invariants()

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

        result = InvariantLearner(
            self.json_grammar,
            prop,
            activated_patterns={"String Existence"},
            positive_examples=trees
        ).learn_invariants()

        # print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            list(map(lambda f: ISLaUnparser(f).unparse(), [r for r, p in result.items() if p == 1.0])))

    def test_load_patterns_from_file(self):
        patterns = patterns_from_file()
        self.assertTrue(patterns)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertIn("Def-Use", patterns)
        self.assertIn("Def-Use (C)", patterns)
        self.assertIn("Def-Use (XML)", patterns)
        self.assertNotIn("Def-Use (...)", patterns)


if __name__ == '__main__':
    unittest.main()
