import copy
import json
import unittest
from typing import cast

from fuzzingbook.Grammars import JSON_GRAMMAR
from fuzzingbook.Parser import EarleyParser
from isla import language
from isla.evaluator import evaluate
from isla.isla_predicates import STANDARD_SEMANTIC_PREDICATES
from isla.language import parse_isla, ISLaUnparser
from isla_formalizations import scriptsizec, csv

from islearn.learner import generate_candidates, patterns_from_file, learn_invariants


class TestLearner(unittest.TestCase):
    def test_filter_invariants_simple_scriptsize_c(self):
        pattern = """
forall <?NONTERMINAL> use_ctx in start:
  forall <?NONTERMINAL> use in use_ctx:
    exists <?NONTERMINAL> def_ctx in start:
      exists <?NONTERMINAL> def in def_ctx:
        (before(def_ctx, use_ctx) and
        (= def use))"""

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        candidates = generate_candidates([pattern], inputs, scriptsizec.SCRIPTSIZE_C_GRAMMAR)
        result = [candidate for candidate in candidates
                  if all(evaluate(candidate, inp, scriptsizec.SCRIPTSIZE_C_GRAMMAR)
                         for inp in inputs)]
        # print("\n\n".join(map(lambda f: ISLaUnparser(f).unparse(), result)))

        self.assertEqual(8, len(result))
        for formula in result:
            vars = {var.name: var for var in language.VariablesCollector.collect(formula)}
            self.assertIn(vars["use_ctx"].n_type, ["<expr>", "<test>", "<sum>", "<term>"])
            self.assertEqual("<id>", vars["use"].n_type)
            self.assertIn(vars["def_ctx"].n_type, ["<block_statement>", "<declaration>"])
            self.assertEqual("<id>", vars["def"].n_type)

    def test_learn_invariants_simple_scriptsize_c(self):
        correct_property = """
forall <expr> use_ctx in start:
  forall <id> use in use_ctx:
    exists <declaration> def_ctx in start:
      exists <id> def in def_ctx:
        (before(def_ctx, use_ctx) and
        (= def use))"""

        def prop(tree: language.DerivationTree) -> bool:
            return evaluate(correct_property, tree, scriptsizec.SCRIPTSIZE_C_GRAMMAR).is_true()

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
            activated_patterns={"Def-Use"},
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
    count(elem, <?NONTERMINAL>, num)"""

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
        result = [candidate for candidate in candidates
                  if all(evaluate(candidate, inp, csv.CSV_GRAMMAR)
                         for inp in inputs)]

        self.assertEqual(2, len(result))
        for formula in result:
            vars = {var.name: var for var in language.VariablesCollector.collect(formula)}
            self.assertEqual(vars["elem"].n_type, "<csv-record>")

            needle_values = {
                cast(language.SemanticPredicateFormula, f).args[1]
                for f in
                language.FilterVisitor(lambda f: isinstance(f, language.SemanticPredicateFormula)).collect(formula)}
            for needle in needle_values:
                self.assertIn(needle, {"<csv-string-list>", "<raw-field>"})

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

        # print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            parse_isla(correct_property, csv.CSV_GRAMMAR, semantic_predicates=STANDARD_SEMANTIC_PREDICATES),
            result)

        self.assertEqual(2, len(result))
        for formula in result:
            vars = {var.name: var for var in language.VariablesCollector.collect(formula)}
            self.assertEqual(vars["elem"].n_type, "<csv-record>")

            needle_values = {
                cast(language.SemanticPredicateFormula, f).args[1]
                for f in
                language.FilterVisitor(lambda f: isinstance(f, language.SemanticPredicateFormula)).collect(formula)}
            for needle in needle_values:
                self.assertIn(needle, {"<csv-string-list>", "<raw-field>"})

    def test_string_existence(self):
        correct_property = """
forall <object> container in start:
  exists <member> inner_container in container:
    exists <string> elem in inner_container:
      (= elem \"""key""\")
"""

        grammar = copy.deepcopy(JSON_GRAMMAR)
        grammar["<value>"] = ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"]
        grammar["<int>"] = ["<digit>", "<onenine><digits>", "-<digit>", "-<onenine><digits>"]

        def prop(tree: language.DerivationTree) -> bool:
            json_obj = json.loads(str(tree))
            return isinstance(json_obj, dict) and "key" in json_obj

        inputs = [' { "key" : 13 } ', ' { "asdf" : [ 13 ] , "key" : "x" } ']
        trees = [language.DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
                 for inp in inputs]

        self.assertTrue(all(evaluate(correct_property, tree, grammar) for tree in trees))

        # candidates = generate_candidates([patterns_from_file()["String Existence"]], trees, grammar)
        # self.assertIn(correct_property.strip(), list(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

        # print(len(candidates))
        # print("\n".join(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

        result = learn_invariants(
            grammar,
            prop,
            activated_patterns={"String Existence"},
            positive_examples=trees
        )

        # print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(correct_property.strip(), list(map(lambda f: ISLaUnparser(f).unparse(), result)))

    def test_load_patterns_from_file(self):
        patterns = patterns_from_file()
        self.assertTrue(patterns)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertIn("Def-Use", patterns)


if __name__ == '__main__':
    unittest.main()
