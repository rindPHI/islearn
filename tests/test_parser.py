import unittest

from fuzzingbook.Grammars import JSON_GRAMMAR
from isla import language
from isla_formalizations import scriptsizec

from islearn.language import parse_abstract_isla, AbstractISLaUnparser


class TestParser(unittest.TestCase):
    def test_defuse_pattern(self):
        pattern = """
forall <?NONTERMINAL> use_ctx in start:
  forall <?NONTERMINAL> use in use_ctx:
    exists <?NONTERMINAL> def_ctx in start:
      exists <?NONTERMINAL> def in def_ctx:
        (before(def_ctx, use_ctx) and
        (= def use))"""

        self.assertEqual(
            pattern.strip(),
            language.ISLaUnparser(parse_abstract_isla(pattern, scriptsizec.SCRIPTSIZE_C_GRAMMAR)).unparse())

    def test_count_pattern(self):
        pattern = """
exists int num:
  forall <?NONTERMINAL> elem in start:
    count(elem, <?NONTERMINAL>, num)"""

        self.assertEqual(
            pattern.strip(),
            language.ISLaUnparser(parse_abstract_isla(pattern, scriptsizec.SCRIPTSIZE_C_GRAMMAR)).unparse())

    def test_string_occurrence(self):
        pattern = """
forall <?NONTERMINAL> container in start:
  exists <?NONTERMINAL> elem in container:
    (= elem <?STRING>)"""

        self.assertEqual(pattern.strip(), AbstractISLaUnparser(parse_abstract_isla(pattern, JSON_GRAMMAR)).unparse())


if __name__ == '__main__':
    unittest.main()
