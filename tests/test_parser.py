import unittest

from isla import language
from isla_formalizations import xml_lang

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
            language.ISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_count_pattern(self):
        pattern = """
exists int num:
  forall <?NONTERMINAL> elem in start:
    count(elem, <?NONTERMINAL>, num)"""

        self.assertEqual(
            pattern.strip(),
            language.ISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_string_occurrence(self):
        pattern = """
forall <?NONTERMINAL> container in start:
  exists <?NONTERMINAL> elem in container:
    (= elem <?STRING>)"""

        self.assertEqual(pattern.strip(), AbstractISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_schematic_match_expression(self):
        pattern = """
forall <?NONTERMINAL> use_ctx in start:
  forall <?NONTERMINAL> use in use_ctx:
    exists <?NONTERMINAL> def_ctx="{<?MATCHEXPR(def)>}" in start:
      (before(def_ctx, use_ctx) and
      (= use def))"""

        self.assertEqual(
            pattern.strip(),
            AbstractISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_partially_instantiated_match_expression(self):
        pattern = """
forall <?NONTERMINAL> use_ctx in start:
  forall <?NONTERMINAL> use in use_ctx:
    exists <?NONTERMINAL> def_ctx="{<?MATCHEXPR(<id> def)>}" in start:
      (before(def_ctx, use_ctx) and
      (= use def))"""

        self.assertEqual(
            pattern.strip(),
            AbstractISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_dstrings_placeholder(self):
        pattern = """
forall <?NONTERMINAL> attribute in start:
  forall <?NONTERMINAL> prefix_id="{<?MATCHEXPR(prefix_use)>}" in attribute:
    ((= prefix_use <?DSTRINGS>) or
    exists <?NONTERMINAL> outer_tag="{<?MATCHEXPR(cont_attribute, contained_tree)>}" in start:
      (inside(attribute, contained_tree) and
      exists <?NONTERMINAL> def_attribute="{<?MATCHEXPR(prefix_def)>}" in cont_attribute:
        (= prefix_use prefix_def)))
"""

        self.assertEqual(
            pattern.strip(),
            AbstractISLaUnparser(parse_abstract_isla(pattern)).unparse())

    def test_parse_match_expressions_with_instantiated_arguments(self):
        parse_abstract_isla("""
forall <xml-tree> xml_tree="{<?MATCHEXPR(<id-no-prefix> prefix_use)>}" in start:
  exists <xml-tree> outer_tag="{<?MATCHEXPR(<xml-attribute> cont_attribute)>}" in start:
    (inside(xml_tree, outer_tag) and
     exists <xml-attribute> def_attribute="{<?MATCHEXPR(<id-no-prefix> ns_prefix, <id-no-prefix> prefix_def)>}" 
         in cont_attribute:
       ((= ns_prefix <?STRING>) and
        (= prefix_use prefix_def)))
""", xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES)

    def test_toml_value_type_is_string(self):
        parse_abstract_isla('''
forall <?NONTERMINAL> container="{<?MATCHEXPR(key, value)>}" in start:
  ((= key <?STRING>) implies
   (str.in_re value (re.++ (str.to_re "")
                           (re.* (re.comp (re.union (str.to_re "\n") (str.to_re ""))))
                           (str.to_re ""))))
''')


if __name__ == '__main__':
    unittest.main()
