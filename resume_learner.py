import logging
import pickle
import time
from typing import List, Dict, Set

from isla import isla

from islearn.learner import collect_nonvacuously_satisfied_candidates, collect_nonvacuously_satisfied_candidates_2, \
    check_vacuous_satisfaction

# with (open("/tmp/saved_debug_state", "rb")) as debug_state_file:
#     logging.basicConfig(level=logging.INFO)
#
#     try:
#         candidates, grammar, inputs = pickle.load(debug_state_file)
#
#         start = time.time()
#         result = collect_nonvacuously_satisfied_candidates(candidates, grammar, inputs)
#         end = time.time()
#         print(f"Time: {end - start} s")
#
#         print("\n\n".join(map(isla.unparse_isla, result)))
#     except EOFError as e:
#         print(e)

with (open("/tmp/saved_debug_state_1", "rb")) as debug_state_file:
    logging.basicConfig(level=logging.INFO)

    try:
        valid_candidates: Dict[isla.Formula, Dict[isla.Formula, Set[isla.ForallFormula]]] = \
            pickle.load(debug_state_file)

        start = time.time()
        nonvacuously_satisfied_candidates: List[isla.Formula] = [
            candidate for candidate in valid_candidates
            if any(not check_vacuous_satisfaction(instantiated_formula, vacuously_matched_quantifiers)
                   for instantiated_formula, vacuously_matched_quantifiers in valid_candidates[candidate].items())
        ]
        end = time.time()
        print(f"Time: {end - start} s")

        print("\n\n".join(map(isla.unparse_isla, nonvacuously_satisfied_candidates)))
    except EOFError as e:
        print(e)
