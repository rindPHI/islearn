import logging
from typing import List, Tuple

from grammar_graph import gg
from isla import isla
from isla.solver import get_quantifier_chains
from isla.type_defs import Grammar

from islearn.pattern import Pattern, Placeholders

logger = logging.getLogger("learner")


def filter_invariants(
        patterns: List[Pattern],
        inputs: List[isla.DerivationTree],
        grammar: Grammar) -> List[isla.Formula]:
    graph = gg.GrammarGraph.from_grammar(grammar)

    # Create all pattern instantiations
    logger.info("Instantiating patterns to grammar.")
    num_insts = sum([
        len(grammar) ** len(pattern.placeholders)
        for pattern in patterns
    ])
    logger.info(f"There are %d instantiations I'll try.", num_insts)

    i = 0
    candidates: List[isla.Formula] = []
    for pattern_idx, pattern in enumerate(patterns):
        logger.debug("Instantiating pattern %d of %d", pattern_idx + 1, len(patterns))

        partially_instantiated_placeholders: List[Placeholders] = [pattern.placeholders]
        while partially_instantiated_placeholders:
            ph = partially_instantiated_placeholders.pop()

            precondition_valid = pattern.precondition(ph, grammar, graph)
            if precondition_valid is False:
                continue

            if ph.next_uninstantiated_placeholder() is None:
                assert precondition_valid is not None
                if precondition_valid:
                    candidates.append(pattern.formula_factory(ph, grammar))

                continue

            # TODO: Numeric constants
            for nonterminal in grammar:
                assert ph.next_uninstantiated_placeholder() is not None
                partially_instantiated_placeholders.append(
                    ph.instantiate(ph.next_uninstantiated_placeholder(), nonterminal))

                i += 1
                if i % int(num_insts / 10) == 0:
                    perc_done = int(i * 100 / num_insts)
                    logger.info("%d%% done", perc_done)

    logger.info("Found %d invariant candidates.", len(candidates))
    logger.info("Checking with %d sample inputs.", len(inputs))

    satisfied_candidates = [
        candidate for candidate in candidates
        if all(isla.evaluate(candidate, inp, grammar) for inp in inputs)
    ]

    logger.info(
        "Found %d invariants, checking nonvacuous satisfaction.",
        len(satisfied_candidates))

    nonvacuously_satisfied_candidates = [
        candidate for candidate in satisfied_candidates
        if not vacuously_satisfied(candidate, inputs, graph)
    ]

    logger.info(
        "Found %d non-vacuously satisfied invariants",
        len(nonvacuously_satisfied_candidates))

    return nonvacuously_satisfied_candidates


def vacuously_satisfied(
        formula: isla.Formula,
        inputs: List[isla.DerivationTree],
        graph: gg.GrammarGraph) -> bool:
    if not isla.get_toplevel_quantified_formulas(formula):
        return False

    constant: isla.Constant = next(
        (c for c in isla.VariablesCollector.collect(formula)
         if isinstance(c, isla.Constant) and not c.is_numeric()))

    for inp in inputs:
        # TODO: Deal with conjunctions / disjunctions and v.s. in only one part.
        instantiated_formula = formula.substitute_expressions({constant: inp})
        isla.set_smt_auto_eval(instantiated_formula, False)

        quantifier_chains: List[Tuple[isla.ForallFormula, ...]] = [
            tuple([f for f in c if isinstance(f, isla.ForallFormula)])
            for c in get_quantifier_chains(instantiated_formula)]
        quantifier_chains = [c for c in quantifier_chains if c]

        vacuously_matched_quantifiers = set()
        isla.eliminate_quantifiers(
            instantiated_formula,
            vacuously_satisfied=vacuously_matched_quantifiers,
            grammar=graph.to_grammar(),
            graph=graph)

        vacuous_chains = {
            c for c in quantifier_chains if
            any(any(of.id == f.id for of in vacuously_matched_quantifiers)
                for f in c)}

        assert len(vacuous_chains) <= len(quantifier_chains)
        if len(vacuous_chains) < len(quantifier_chains):
            return False

    return True
