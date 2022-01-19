import unittest
from typing import cast

import z3
from fuzzingbook.Parser import EarleyParser
from isla import isla, isla_shortcuts as sc

from islearn.learner import filter_invariants, NonterminalPlaceholderVariable
from islearn.tests.subject_languages import scriptsizec


class TestLearner(unittest.TestCase):
    def test_filter_invariants_simple_scriptsize_c(self):
        start = isla.Constant("start", "<start>")
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
                        isla.SMTFormula(cast(z3.BoolRef, def_.to_smt() == use.to_smt()), def_, use) &
                        sc.before(def_ctx, use_ctx)
                    ))))

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            isla.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        result = filter_invariants([abstract_formula], inputs, scriptsizec.SCRIPTSIZE_C_GRAMMAR)

        self.assertEqual(8, len(result))
        for formula in result:
            vars = {var.name: var for var in isla.VariablesCollector.collect(formula)}
            self.assertIn(vars["use_ctx"].n_type, ["<expr>", "<test>", "<sum>", "<term>"])
            self.assertEqual("<id>", vars["use"].n_type)
            self.assertIn(vars["def_ctx"].n_type, ["<block_statement>", "<declaration>"])
            self.assertEqual("<id>", vars["def"].n_type)


if __name__ == '__main__':
    unittest.main()
