import logging
from typing import List, Tuple, Set, Dict

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

    nonterminals_in_inputs = {nonterminal for inp in inputs for nonterminal in inp.nonterminals()}

    # Create all pattern instantiations
    logger.info("Instantiating patterns to grammar.")
    num_insts = sum([
        len(nonterminals_in_inputs) ** len(pattern.placeholders)
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
            for nonterminal in nonterminals_in_inputs:
                assert ph.next_uninstantiated_placeholder() is not None
                partially_instantiated_placeholders.append(
                    ph.instantiate(ph.next_uninstantiated_placeholder(), nonterminal))

                i += 1
                if i % int(num_insts / 10) == 0:
                    perc_done = int(i * 100 / num_insts)
                    logger.info("%d%% done", perc_done)

    logger.info("Found %d invariant candidates.", len(candidates))
    logger.info("Checking with %d sample inputs.", len(inputs))

    # satisfied_candidates = [
    #     candidate for candidate in candidates
    #     if all(isla.evaluate(candidate, inp, grammar) for inp in inputs)
    # ]
    #
    # logger.info(
    #     "Found %d invariants, checking nonvacuous satisfaction.",
    #     len(satisfied_candidates))
    #
    # nonvacuously_satisfied_candidates = [
    #     candidate for candidate in satisfied_candidates
    #     if not vacuously_satisfied(candidate, inputs, graph)
    # ]

    # import dill as pickle
    # out_file = "/tmp/saved_debug_state"
    # with open(out_file, 'wb') as debug_state_file:
    #     pickle.dump((candidates, grammar, inputs), debug_state_file)
    # print(f"Dumping state to {out_file}")
    # exit()

    return collect_nonvacuously_satisfied_candidates(candidates, grammar, inputs)


def collect_nonvacuously_satisfied_candidates_2(candidates, grammar, inputs):
    graph = gg.GrammarGraph.from_grammar(grammar)

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

    return nonvacuously_satisfied_candidates


def collect_nonvacuously_satisfied_candidates(candidates, grammar, inputs):
    i = 0
    valid_candidates: Dict[isla.Formula, Dict[isla.Formula, Set[isla.ForallFormula]]] = {}
    for candidate in candidates:
        constant: isla.Constant = extract_top_level_constant(candidate)

        vacuously_matched_quantifier_map: Dict[isla.Formula, Set[isla.ForallFormula]] = {}
        for inp in inputs:
            if i % int((len(candidates) * len(inputs)) / 10) == 0:
                perc_done = int(i * 100 / (len(candidates) * len(inputs)))
                logger.info("%d%% done", perc_done)
            i += 1

            instantiated_formula = candidate.substitute_expressions({constant: inp})

            vacuously_matched_quantifiers = set()
            if not isla.evaluate(instantiated_formula, inp, grammar, vacuously_satisfied=vacuously_matched_quantifiers):
                break

            vacuously_matched_quantifier_map[instantiated_formula] = vacuously_matched_quantifiers
        else:
            valid_candidates[candidate] = vacuously_matched_quantifier_map

    logger.info("Found %d invariants", len(valid_candidates))

    # import dill as pickle
    # out_file = "/tmp/saved_debug_state_1"
    # with open(out_file, 'wb') as debug_state_file:
    #     pickle.dump(valid_candidates, debug_state_file)
    # print(f"Dumping state to {out_file}")
    # exit()

    nonvacuously_satisfied_candidates: List[isla.Formula] = [
        candidate for candidate in valid_candidates
        if any(not check_vacuous_satisfaction(instantiated_formula, vacuously_matched_quantifiers)
               for instantiated_formula, vacuously_matched_quantifiers in valid_candidates[candidate].items())
    ]

    logger.info(
        "Found %d non-vacuously satisfied invariants",
        len(nonvacuously_satisfied_candidates))

    return nonvacuously_satisfied_candidates


def extract_top_level_constant(candidate):
    return next(
        (c for c in isla.VariablesCollector.collect(candidate)
         if isinstance(c, isla.Constant) and not c.is_numeric()))


def check_vacuous_satisfaction(
        formula: isla.Formula,
        vacuously_matched_quantifiers: Set[isla.ForallFormula]) -> bool:
    if not isla.get_toplevel_quantified_formulas(formula) or not vacuously_matched_quantifiers:
        return False

    # TODO: Deal with conjunctions / disjunctions and v.s. in only one part.
    quantifier_chains: List[Tuple[isla.ForallFormula, ...]] = [
        tuple([f for f in c if isinstance(f, isla.ForallFormula)])
        for c in get_quantifier_chains(formula)]
    quantifier_chains = [c for c in quantifier_chains if c]

    vacuous_chains = {
        c for c in quantifier_chains if
        any(any(of.id == f.id for of in vacuously_matched_quantifiers)
            for f in c)}

    assert len(vacuous_chains) <= len(quantifier_chains)
    if len(vacuous_chains) < len(quantifier_chains):
        return False

    return True


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
