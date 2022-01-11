import unittest
from typing import Optional

from fuzzingbook.Parser import EarleyParser
from grammar_graph import gg
from grammar_graph.gg import Grammar
from isla import isla, isla_shortcuts as sc
from isla.isla_predicates import reachable

from islearn.learner import filter_invariants, vacuously_satisfied
from islearn.pattern import Pattern, Placeholders, Placeholder
from islearn.tests.subject_languages import scriptsizec


class TestLearner(unittest.TestCase):
    def test_simple_scriptsize_c(self):
        placeholders = Placeholders(
            Placeholder("def", Placeholder.PlaceholderType.NONTERMINAL),
            Placeholder("def_ctx", Placeholder.PlaceholderType.NONTERMINAL),
            Placeholder("use", Placeholder.PlaceholderType.NONTERMINAL),
            Placeholder("use_ctx", Placeholder.PlaceholderType.NONTERMINAL),
        )

        def precondition(placeholders: Placeholders, _, graph: gg.GrammarGraph) -> Optional[bool]:
            use_ctx = placeholders["use_ctx"]
            use = placeholders["use"]
            def_ctx = placeholders["def_ctx"]
            def_ = placeholders["def"]

            r1 = None
            r2 = None

            if use_ctx is not None and use is not None:
                r1 = reachable(graph, use_ctx, use)

            if def_ctx is not None and use_ctx is not None:
                r2 = reachable(graph, def_ctx, def_)

            if r1 is False or r2 is False:
                return False

            if r1 is True and r2 is True:
                return True

            assert r1 is None or r2 is None
            return None

        def formula_factory(placeholders: Placeholders, grammar: Grammar) -> isla.Formula:
            mgr = isla.VariableManager(grammar)
            return mgr.create(
                sc.forall(
                    mgr.bv("use_ctx", placeholders["use_ctx"]),
                    mgr.const("start", "<start>"),
                    sc.forall(
                        mgr.bv("use", placeholders["use"]),
                        mgr.bv("use_ctx"),
                        sc.exists(
                            mgr.bv("def_ctx", placeholders["def_ctx"]),
                            mgr.const("start"),
                            sc.exists(
                                mgr.bv("def", placeholders["def"]),
                                mgr.bv("def_ctx"),
                                mgr.smt(mgr.bv("def").to_smt() == mgr.bv("use").to_smt()) &
                                sc.before(mgr.bv("def_ctx"), mgr.bv("use_ctx"))
                            )
                        )
                    )
                )
            )

        pattern = Pattern(placeholders, precondition, formula_factory)
        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            isla.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(input)))
            for input in raw_inputs]

        result = filter_invariants([pattern], inputs, scriptsizec.SCRIPTSIZE_C_GRAMMAR)

        print("\n\n".join(map(repr, result)))

    def test_vacuous_satisfaction(self):
        grammar = scriptsizec.SCRIPTSIZE_C_GRAMMAR

        mgr = isla.VariableManager(scriptsizec.SCRIPTSIZE_C_GRAMMAR)
        formula = mgr.create(
            sc.forall(
                mgr.bv("use_ctx", "<block>"),
                mgr.const("start", "<start>"),
                sc.forall(
                    mgr.bv("use", "<paren_expr>"),
                    mgr.bv("use_ctx"),
                    sc.exists(
                        mgr.bv("def_ctx", "<sum>"),
                        mgr.const("start"),
                        sc.exists(
                            mgr.bv("def", "<digit_nonzero>"),
                            mgr.bv("def_ctx"),
                            mgr.smt(mgr.bv("def").to_smt() == mgr.bv("use").to_smt()) &
                            sc.before(mgr.bv("def_ctx"), mgr.bv("use_ctx"))
                        )))))

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            isla.DerivationTree.from_parse_tree(
                next(EarleyParser(grammar).parse(inp)))
            for inp in raw_inputs]

        self.assertTrue(vacuously_satisfied(formula, inputs, gg.GrammarGraph.from_grammar(grammar)))


if __name__ == '__main__':
    unittest.main()
