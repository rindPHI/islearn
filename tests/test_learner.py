import unittest
from typing import cast

import z3
from fuzzingbook.Parser import EarleyParser
from isla import language, isla_shortcuts as sc
from isla_formalizations import scriptsizec, csv

from islearn.learner import filter_invariants, NonterminalPlaceholderVariable, NonterminalStringPlaceholderVariable


class TestLearner(unittest.TestCase):
    def test_filter_invariants_simple_scriptsize_c(self):
        start = language.Constant("start", "<start>")
        use_ctx = NonterminalPlaceholderVariable("use_ctx")
        use = NonterminalPlaceholderVariable("use")
        def_ctx = NonterminalPlaceholderVariable("def_ctx")
        def_ = NonterminalPlaceholderVariable("def")

        abstract_formula = sc.forall(
            use_ctx,
            start,
            sc.forall(
                use,
                use_ctx,
                sc.exists(
                    def_ctx,
                    start,
                    sc.exists(
                        def_,
                        def_ctx,
                        language.SMTFormula(cast(z3.BoolRef, def_.to_smt() == use.to_smt()), def_, use) &
                        sc.before(def_ctx, use_ctx)
                    ))))

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        result = filter_invariants([abstract_formula], inputs, scriptsizec.SCRIPTSIZE_C_GRAMMAR)
        print("\n\n".join(map(language.unparse_isla, result)))

        self.assertEqual(8, len(result))
        for formula in result:
            vars = {var.name: var for var in language.VariablesCollector.collect(formula)}
            self.assertIn(vars["use_ctx"].n_type, ["<expr>", "<test>", "<sum>", "<term>"])
            self.assertEqual("<id>", vars["use"].n_type)
            self.assertIn(vars["def_ctx"].n_type, ["<block_statement>", "<declaration>"])
            self.assertEqual("<id>", vars["def"].n_type)

    def test_filter_invariants_simple_csv_colno(self):
        start = language.Constant("start", "<start>")
        elem = NonterminalPlaceholderVariable("elem")
        num = language.BoundVariable("num", language.Constant.NUMERIC_NTYPE)
        needle_placeholder = NonterminalStringPlaceholderVariable("needle")

        abstract_formula = language.ExistsIntFormula(
            num,
            sc.forall(
                elem,
                start,
                sc.count({}, elem, needle_placeholder, num)
            )
        )

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

        result = filter_invariants([abstract_formula], inputs, csv.CSV_GRAMMAR)
        print(len(result))
        print("\n".join(map(language.unparse_isla, result)))

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

    def test_filter_invariants_complex_csv_colno(self):
        # TODO
        start = language.Constant("start", "<start>")
        elem = NonterminalPlaceholderVariable("elem")
        num = language.BoundVariable("num", language.Constant.NUMERIC_NTYPE)
        needle_placeholder = NonterminalStringPlaceholderVariable("needle")

        abstract_formula = language.ExistsIntFormula(
            num,
            sc.forall(
                elem,
                start,
                sc.count({}, elem, needle_placeholder, num)
            )
        )

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

        result = filter_invariants([abstract_formula], inputs, csv.CSV_GRAMMAR)
        print(len(result))
        print("\n".join(map(language.unparse_isla, result)))

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


if __name__ == '__main__':
    unittest.main()
