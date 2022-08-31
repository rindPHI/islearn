import copy
import json
import math
import random
import re
import string
import sys
import unittest
import urllib.request
from typing import cast, Tuple, Set

import pytest
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.helpers import strip_ws, srange
from isla.isla_predicates import STANDARD_SEMANTIC_PREDICATES, BEFORE_PREDICATE, IN_TREE_PREDICATE
from isla.language import parse_isla, ISLaUnparser
from isla.parser import EarleyParser
from isla.parser import PEGParser
from isla.solver import implies, equivalent
from isla_formalizations import scriptsizec, csv, xml_lang, rest
from isla_formalizations.csv import CSV_HEADERBODY_GRAMMAR
from pythonping import icmp

from islearn.islearn_predicates import hex_to_bytes, bytes_to_hex
from islearn.language import parse_abstract_isla, NonterminalPlaceholderVariable, ISLEARN_STANDARD_SEMANTIC_PREDICATES, \
    AbstractISLaUnparser, unparse_abstract_isla
from islearn.learner import patterns_from_file, InvariantLearner, \
    create_input_reachability_relation, InVisitor, approximately_evaluate_abst_for, PatternRepository
from islearn_example_languages import toml_grammar, JSON_GRAMMAR, ICMP_GRAMMAR, IPv4_GRAMMAR, DOT_GRAMMAR, render_dot, \
    RACKET_BSL_GRAMMAR, load_racket


class TestLearner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_instantiate_nonterminal_placeholders_scriptsize_c(self):
        expected = """
forall <expr> use_ctx in start:
  forall <id> use in use_ctx:
    exists <declaration> def_ctx="{<?MATCHEXPR(<id> def)>}" in start:
      (before(def_ctx, use_ctx) and
      (= use def))"""

        learner = InvariantLearner(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            activated_patterns={"Def-Use (C)"},
        )

        good_programs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        good_inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in good_programs]

        instantiations = learner._instantiate_nonterminal_placeholders(
            next(iter(patterns_from_file()["Def-Use (C)"])),
            create_input_reachability_relation(good_inputs)
        )

        print("\n".join(map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations)))
        self.assertIn(parse_abstract_isla(expected.strip()), instantiations)

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

        good_programs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        good_inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in good_programs]

        # bad_programs = [
        #     "{int c;d < 0;}",
        #     "{17 < 0;x = y + 1;}",
        # ]
        # bad_inputs = [
        #     language.DerivationTree.from_parse_tree(
        #         next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
        #     for inp in bad_programs]

        #########
        # candidates = InvariantLearner(
        #     scriptsizec.SCRIPTSIZE_C_GRAMMAR,
        #     prop,
        #     activated_patterns={"Def-Use (C)"},
        #     positive_examples=good_inputs
        # ).generate_candidates(
        #     patterns_from_file()["Def-Use (C)"],
        #     good_inputs)
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # self.assertIn(
        #     correct_property.strip(),
        #     map(lambda f: ISLaUnparser(f).unparse(), candidates))
        #
        # return
        ##########

        result = InvariantLearner(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            prop,
            activated_patterns={"Def-Use (C)"},
            positive_examples=good_inputs,
            target_number_positive_samples=7,
            target_number_positive_samples_for_learning=4,
            max_conjunction_size=1,
        ).learn_invariants()

        # print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_learn_invariants_mexpr_rest(self):
        correct_property = """
forall <internal_reference> use_ctx="<presep>{<id> use}_<postsep>" in start:
  exists <labeled_paragraph> def_ctx=".. _{<id> def}:\n\n<paragraph>" in start:
    (different_position(use_ctx, def_ctx) and
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
        # candidates = InvariantLearner(
        #     rest.REST_GRAMMAR,
        #     prop,
        #     activated_patterns={"Def-Use (reST)"},
        #     mexpr_expansion_limit=2
        # ).generate_candidates(
        #     patterns_from_file()["Def-Use (reST)"],
        #     inputs,
        # )
        #
        # print(len(candidates))
        # print("\n\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##########

        result = InvariantLearner(
            rest.REST_GRAMMAR,
            prop,
            activated_patterns={"Def-Use (reST)"},
            positive_examples=inputs,
            target_number_positive_samples=7,
            target_number_positive_samples_for_learning=4,
            mexpr_expansion_limit=2,
            k=4,  # TODO: Consider *all* k-paths *up to* 4?
            max_conjunction_size=1,
            min_specificity=.0,
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    # @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_learn_invariants_mexpr_xml(self):
        correct_property = """
forall <xml-tree> container="<{<id> opid}><inner-xml-tree></{<id> clid}>" in start:
  (= opid clid)"""

        def prop(tree: language.DerivationTree) -> bool:
            return xml_lang.validate_xml(tree) is True

        raw_inputs = [
            "<a>asdf</a>",
            "<b><f>xyz</f><c/><x>X</x></b>",
            "<a/>"
        ]

        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(xml_lang.XML_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        ##########
        # candidates = InvariantLearner(
        #     xml_lang.XML_GRAMMAR,
        #     prop,
        #     activated_patterns={"Balance"},
        #     positive_examples=inputs
        # ).generate_candidates(
        #     patterns_from_file()["Balance"],
        #     inputs
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
            positive_examples=inputs,
            mexpr_expansion_limit=2,
            min_specificity=.3  # Low precision needed because inv holds trivially for self-closing tags
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
        print("\n".join(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

        self.assertIn(correct_property.strip(), list(map(lambda f: ISLaUnparser(f).unparse(), candidates)))

    def test_learn_invariants_simple_csv_colno(self):
        correct_property = """
exists int num:
  forall <csv-record> elem in start:
    ((>= (str.to.int num) 1) and
    count(elem, "<raw-field>", num))""".strip()

        def prop(tree: language.DerivationTree) -> bool:
            return evaluate(correct_property, tree, csv.CSV_GRAMMAR).is_true()

        #################
        # inputs = list(map(
        #     lambda inp: language.DerivationTree.from_parse_tree(next(EarleyParser(csv.CSV_GRAMMAR).parse(inp))),
        #     ["a;b;c\n", "a;b\nc;d\n"]))
        #
        # result = InvariantLearner(
        #     csv.CSV_GRAMMAR,
        #     prop,
        #     activated_patterns={"Equal Count"},
        # ).generate_candidates(patterns_from_file()["Equal Count"], inputs)
        #
        # print(len(result))
        # print("\n".join(map(lambda f: ISLaUnparser(f).unparse(), result)))
        #
        # self.assertIn(correct_property.strip(), [ISLaUnparser(f).unparse() for f in result])
        #
        # return
        ##################

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

        perfect_precision_formulas = [f for f, p in result.items() if p == (1.0, 1.0)]
        # self.assertEqual(2, len(perfect_precision_formulas))
        self.assertIn(correct_property, [ISLaUnparser(f).unparse() for f in perfect_precision_formulas])

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_string_existence(self):
        correct_property = r'''
forall <json> container in start:
  exists <string> elem in container:
    (= elem "\"key\"")
'''

        def prop(tree: language.DerivationTree) -> bool:
            json_obj = json.loads(str(tree))
            return isinstance(json_obj, dict) and "key" in json_obj

        inputs = [
            ' { "key" : 13 } ',
            ' { "asdf" : [ 26 ] , "key" : "x" } ',
        ]

        # inputs = [
        #     ' { "asdf?" : [ 26 ] , "key" : "x" } ',
        #     ' { "asdf" : [ 26 ] , "key" : "x" } ',
        #     ' { "key" : { "key" : false } , "" : false , "lR" : true , "" : null , "lR" : { "key" : { "." : [ -0 , false , true , null , false ] } } } ',
        #     ' { "key" : { "key" : 16 } , "" : false , "ld" : true , "" : { } , "lR" : { "" : null } } ',
        #     ' { "key" : { "key" : { "." : [ -0 , false , true , null , false ] } , "" : false , "lR" : true , "" : null , "lR" : { "" : null } } , "ey" : { "" : null } } ',
        #     ' { "key" : { "key" : { "." : [ -0 , false , true , null , false ] } , "" : false , "lR" : true , "" : null , "lR" : { "" : null } } , "ey" : { "kee" : null } } ',
        #     ' { "key" : { "sey" : { "key" : 13 } , "lsdf" : false , "2R" : true , "" : null , "" : { "z" : 16 } } , "" : "x" , "lR" : true , "d" : null , "ey" : { "kd" : { "asdR?" : "e" , "?ey" : "x" } } } ',
        #     ' { "key" : { "kee" : { "asdR?" : "x" , "rey" : "x" } , "" : "x" , "lR" : true , "d" : 10 , "e" : { "kee" : { "" : "" } } } , "" : { "6v$." : false } , "lR" : "h" , "" : null , "ey" : { "" : null } } ',
        #     ' { "key" : { "kXy" : { "key" : 13 } , "lsdf" : false , "l(" : { "Nr" : true , "asdR?" : "x" , "" : false } , "" : null , "lR" : { "q" : null } } , "" : false , "kke" : false , "" : null , "i" : { "key" : { "asdRx" : "dRx" , "kei" : "x" } } } ',
        #     ' { "key" : { "key" : { "." : [ 10 , false , true , null , false ] } , "." : false , "q" : "du" , "" : null , "lR" : { "x" : { "" : null } , "" : false , "" : null , "" : { "lR" : { "asdf" : null , "lR" : true , "" : { "lR" : null } , "lR" : { "" : null } } } , "key" : { "key" : 16 } } } , "ey" : { "kee" : null } } '
        # ]

        trees = [language.DerivationTree.from_parse_tree(PEGParser(JSON_GRAMMAR).parse(inp)[0])
                 for inp in inputs]

        ###############
        # candidates = InvariantLearner(
        #     JSON_GRAMMAR,
        # ).generate_candidates(
        #     patterns_from_file()["String Existence"],
        #     trees,
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ###############

        self.assertTrue(all(evaluate(correct_property, tree, JSON_GRAMMAR) for tree in trees))

        result = InvariantLearner(
            JSON_GRAMMAR,
            prop,
            activated_patterns={"String Existence"},
            positive_examples=trees,
            max_conjunction_size=1
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            list(map(lambda f: ISLaUnparser(f).unparse(), [r for r, p in result.items() if p[0] > .0])))

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_alhazen_sqrt_example(self):
        correct_property_1_a = """
(forall <arith_expr> container in start:
   exists <function> elem in container:
     (= elem "sqrt") and
forall <arith_expr> container_0 in start:
  exists <maybe_minus> elem_0 in container_0:
    (= elem_0 "-"))"""

        correct_property_1_b = """
(forall <arith_expr> container in start:
   exists <maybe_minus> elem in container:
     (= elem "-") and
forall <arith_expr> container_0 in start:
  exists <function> elem_0 in container_0:
    (= elem_0 "sqrt"))"""

        correct_property_2_a_re = re.escape("""
(forall <arith_expr> container in start:
   exists <function> elem in container:
     (= elem "sqrt") and
forall <arith_expr> container_0 in start:
  exists <number> elem_0 in container_0:
    (<= (str.to.int elem_0) (str.to.int "-""".strip()) + r"[1-9][0-9]*" + re.escape('")))')

        correct_property_2_b_re = re.escape("""
(forall <arith_expr> container in start:
   exists <number> elem in container:
     (<= (str.to.int elem) (str.to.int "-""".strip()) + r"[1-9][0-9]*" + '"' + re.escape(""")) and
forall <arith_expr> container_0 in start:
  exists <function> elem_0 in container_0:
    (= elem_0 "sqrt"))""")

        grammar = {
            "<start>": ["<arith_expr>"],
            "<arith_expr>": ["<function>(<number>)"],
            "<function>": ["sqrt", "sin", "cos", "tan"],
            "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
            "<maybe_minus>": ["", "-"],
            "<onenine>": [str(num) for num in range(1, 10)],
            "<digit>": srange(string.digits),
            "<maybe_digits>": ["", "<digits>"],
            "<digits>": ["<digit>", "<digit><digits>"],
            "<maybe_frac>": ["", ".<digits>"]
        }

        def arith_eval(inp: language.DerivationTree) -> float:
            return eval(str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan})

        def prop(inp: language.DerivationTree) -> bool:
            try:
                arith_eval(inp)
                return False
            except ValueError:
                return True

        inputs = ["sqrt(-2)"]
        trees = [language.DerivationTree.from_parse_tree(next(EarleyParser(grammar).parse(inp)))
                 for inp in inputs]

        #############
        # repo = patterns_from_file()
        # candidates = InvariantLearner(
        #     grammar,
        #     prop,
        #     positive_examples=trees
        # ).generate_candidates(
        #     # list(repo["String Existence"] | (repo["Existence Numeric String Smaller Than"])),
        #     repo["Existence Numeric String Smaller Than"],
        #     trees
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        #############

        self.assertTrue(all(evaluate(correct_property_1_a, tree, grammar) for tree in trees))

        result = InvariantLearner(
            grammar,
            prop,
            activated_patterns={"Existence Numeric String Smaller Than", "String Existence"},
            # activated_patterns={"String Existence"},
            positive_examples=trees,
            min_recall=1.0,
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        nonzero_precision_results = list(map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p[0] > .0]))

        self.assertTrue((correct_property_1_a.strip() in nonzero_precision_results) or
                        (correct_property_1_b.strip() in nonzero_precision_results))

        self.assertTrue(
            any(re.match(correct_property_2_a_re, r) for r in nonzero_precision_results) or
            any(re.match(correct_property_2_b_re, r) for r in nonzero_precision_results)
        )

    def test_learn_from_islearn_patterns_file(self):
        correct_property_1 = """
forall <document> container in start:
  exists <key> elem in container:
    (= elem "name")"""

        correct_property_2 = """
forall <document> container in start:
  exists <key> elem in container:
    (= elem "constraint")"""

        correct_property_3 = """
forall <key> elem in start:
  (>= (str.len elem) (str.to.int "4"))"""

        # islearn_repo_content = pkgutil.get_data("islearn", STANDARD_PATTERNS_REPO).decode("UTF-8").strip()

        repo = patterns_from_file()
        # Only retain one constraint per group; too slow otherwise
        repo.groups = {
            g: dict([cast(Tuple[str, language.Formula], list(constraints.items())[0])])
            for g, constraints in repo.groups.items()}

        trees = []
        for i in range(len(repo.groups)):
            new_repo = copy.deepcopy(repo)
            new_repo.groups = dict([list(repo.groups.items())[i]])
            tree = language.DerivationTree.from_parse_tree(list(PEGParser(toml_grammar).parse(str(new_repo)))[0])
            trees.append(tree)

        ##############
        # repo = patterns_from_file()
        # candidates = InvariantLearner(
        #     toml_grammar,
        #     None,
        #     positive_examples=trees
        # ).generate_candidates(
        #     repo["String Existence"] | repo["Universal"],  # repo["Existence Strings Relative Order"]
        #     trees
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # self.assertIn(correct_property_1.strip(), [ISLaUnparser(c).unparse() for c in candidates])
        # self.assertIn(correct_property_2.strip(), [ISLaUnparser(c).unparse() for c in candidates])
        # self.assertIn(correct_property_3.strip(), [ISLaUnparser(c).unparse() for c in candidates])
        # self.assertIn(correct_property_4.strip(), [ISLaUnparser(c).unparse() for c in candidates])
        #
        # return
        ##############

        result = InvariantLearner(
            toml_grammar,
            prop=None,
            activated_patterns={"String Existence", "Universal"},
            positive_examples=trees,
            reduce_all_inputs=True,
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        nonzero_precision_results = list(map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p[0] > .0]))

        self.assertIn(correct_property_1.strip(), nonzero_precision_results)
        self.assertIn(correct_property_2.strip(), nonzero_precision_results)
        self.assertIn(correct_property_3.strip(), nonzero_precision_results)

    def test_str_len_ph_instantiation(self):
        sys.setrecursionlimit(1500)
        repo = patterns_from_file()
        group = 'Universal'
        repo = PatternRepository({
            group: [
                {'name': name, 'constraint': unparse_abstract_isla(constraint)}
                for name, constraint
                in repo.groups[group].items()]})
        tree = language.DerivationTree.from_parse_tree(PEGParser(toml_grammar).parse(str(repo))[0])

        pattern = parse_abstract_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int <?STRING>))""")

        expected = language.parse_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int "10"))""", toml_grammar)

        result = InvariantLearner(
            toml_grammar,
            prop=None,
        )._instantiate_string_placeholders(
            {pattern}, [tree.trie().trie]
        )

        # print(len(result))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), result)))

        self.assertEqual(1, len(result))
        self.assertIn(expected, result)

    def test_str_len_ph_instantiations(self):
        sys.setrecursionlimit(1500)
        repo = patterns_from_file()
        group = 'Universal'
        repo = PatternRepository({
            group: [
                {'name': name, 'constraint': unparse_abstract_isla(constraint)}
                for name, constraint
                in repo.groups[group].items()]})
        tree = language.DerivationTree.from_parse_tree(PEGParser(toml_grammar).parse(str(repo))[0])

        pattern = parse_abstract_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int <?STRING>))""")

        result = InvariantLearner(
            toml_grammar,
            prop=None,
        )._get_string_placeholder_instantiations(
            {pattern}, [tree.trie().trie]
        )

        self.assertEqual(1, len(result))
        self.assertEqual(1, len(list(result.values())[0]))
        insts: Set[str] = list(list(result.values())[0].values())[0]

        self.assertIn("name", insts)
        self.assertIn("constraint", insts)
        self.assertIn("10", insts)
        self.assertIn("Universal", insts)

    def test_instantiate_nonterminal_placeholders_toml(self):
        graph = gg.GrammarGraph.from_grammar(toml_grammar)
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(toml_grammar).parse("b = 1988-10-09"))[0])
        learner = InvariantLearner(toml_grammar, None, )
        pattern = list(patterns_from_file()["Value Type is Date (TOML)"])[0]

        variable_instantiations = learner._instantiations_for_placeholder_variables(
            pattern,
            create_input_reachability_relation([tree])
        )

        for variable_instantiation in variable_instantiations:
            container = NonterminalPlaceholderVariable("container")
            key = NonterminalPlaceholderVariable("key")
            value = NonterminalPlaceholderVariable("value")
            container_ntype = variable_instantiation[container].n_type
            key_ntype = variable_instantiation[key].n_type
            value_ntype = variable_instantiation[value].n_type

            self.assertTrue(graph.reachable(
                container_ntype,
                key_ntype),
                f"Key nonterminal {key_ntype} not reachable from container nonterminal {container_ntype}")
            self.assertTrue(graph.reachable(
                container_ntype,
                value_ntype),
                f"Value nonterminal {value_ntype} not reachable from container nonterminal {container_ntype}")

        formula_instantiations = learner._instantiate_nonterminal_placeholders(
            pattern, set([]), variable_instantiations)

        for inst in formula_instantiations:
            v = InVisitor()
            inst.accept(v)
            wrong_pairs = [(v1, v2) for v1, v2 in v.result
                           if v1.n_type != v2.n_type and
                           not graph.reachable(v2.n_type, v1.n_type)]
            self.assertFalse(wrong_pairs)

    def test_toml_value_types(self):
        content = '''
name = "Dominic Steinhoefel"
birthdate = 1988-10-09
number = 17
preferred_color = "blue"
paper_finished = 0.3245'''

        tree = language.DerivationTree.from_parse_tree(list(PEGParser(toml_grammar).parse(content))[0])

        expected_constraint_1 = '''
forall <key_value> container="{<key> key} = {<value> value}" in start:
  ((not (= key "birthdate")) or
  (str.in_re 
    value 
    (re.++ 
      (re.++ 
        (re.++ 
          (re.++ 
            ((_ re.loop 4 4) (re.range "0" "9"))
            (str.to_re "-"))
          ((_ re.loop 2 2) (re.range "0" "9")))
        (str.to_re "-"))
      ((_ re.loop 2 2) (re.range "0" "9")))))'''

        expected_constraint_2 = r'''
forall <key_value> container="{<key> key} = {<value> value}" in start:
  ((not (= key "name")) or
  (str.in_re 
    value 
    (re.++ 
      (re.++ 
        (str.to_re "\"")
        (re.* (re.comp (re.union (str.to_re "\n") (str.to_re "\""))))) 
      (str.to_re "\""))))'''

        expected_constraint_3 = '''
forall <key_value> container="{<key> key} = {<value> value}" in start:
  ((not (= key "number")) or
  (str.in_re value (re.++ (re.opt (str.to_re "-")) (re.+ (re.range "0" "9")))))'''

        expected_constraint_4 = '''
forall <key_value> container="{<key> key} = {<value> value}" in start:
  ((not (= key "paper_finished")) or  
  (str.in_re value
     (re.++
       (re.opt (re.union (str.to_re "+") (str.to_re "-")))
       (re.union
         (re.++
           (re.+ (re.range "0" "9"))
           (re.++ (str.to_re ".") (re.* (re.range "0" "9"))))
         (re.++
           (str.to_re ".")
           (re.+ (re.range "0" "9")))))))'''

        ##############
        # repo = patterns_from_file()
        # candidates = InvariantLearner(
        #     toml_grammar,
        #     None,
        #     mexpr_expansion_limit=2
        # ).generate_candidates(
        #     repo["Value Type is Float (TOML)"],
        #     [tree]
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # self.assertIn(
        #     strip_ws(expected_constraint_4),
        #     list(map(strip_ws, map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        # )
        #
        # return
        ##############

        result = InvariantLearner(
            toml_grammar,
            prop=None,
            activated_patterns={
                "Value Type is Date (TOML)",
                "Value Type is Integer (TOML)",
                "Value Type is String (TOML)",
                "Value Type is Float (TOML)",
            },
            positive_examples=[tree]
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        nonzero_precision_results = list(map(strip_ws, map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p[0] > .0])))

        self.assertIn(strip_ws(expected_constraint_1), nonzero_precision_results)
        self.assertIn(strip_ws(expected_constraint_2), nonzero_precision_results)
        self.assertIn(strip_ws(expected_constraint_3), nonzero_precision_results)
        self.assertIn(strip_ws(expected_constraint_4), nonzero_precision_results)

    def test_csv_value_types(self):
        content = '''Idx;Date;Name;Perc
1;2022-02-28;"Dominic Steinhoefel";.1
17;2021-12-01;"John Doe";17.0
'''
        expected_constraint_1 = """
forall <csv-body> container in start:
  forall <csv-record> row in container:
    exists <raw-field> column in row:
      (nth("1", column, row) and
      (str.in_re column (re.++ (re.opt (str.to_re "-")) (re.+ (re.range "0" "9")))))"""

        tree = language.DerivationTree.from_parse_tree(list(EarleyParser(CSV_HEADERBODY_GRAMMAR).parse(content))[0])

        ##############
        # repo = patterns_from_file()
        # candidates = InvariantLearner(
        #     CSV_HEADERBODY_GRAMMAR,
        #     None,
        # ).generate_candidates(
        #     repo["Value Type is Integer (CSV)"],
        #     [tree]
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # self.assertIn(
        #     strip_ws(expected_constraint_1),
        #     list(map(strip_ws, map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        # )
        #
        # return
        ##############

        result = InvariantLearner(
            CSV_HEADERBODY_GRAMMAR,
            prop=None,
            activated_patterns={
                "Value Type is Integer (CSV)",
                "Value Type is Float (CSV)",
                "Value Type is String (CSV)",
                "Value Type is Date (CSV)",
            },
            positive_examples=[tree]
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        nonzero_precision_results = list(map(strip_ws, map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p[0] > .0])))

        self.assertIn(strip_ws(expected_constraint_1), nonzero_precision_results)

    def test_evaluation_c_defuse(self):
        property = parse_isla("""
forall <expr> use_ctx in start:
  forall <id> use in use_ctx:
    exists <declaration> def_ctx="int {<id> def};" in start:
      (before(def_ctx, use_ctx) and
      (= use def))""", scriptsizec.SCRIPTSIZE_C_GRAMMAR, structural_predicates={BEFORE_PREDICATE})

        inp = language.DerivationTree.from_parse_tree(
            next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse("{int c;c < 0;}")))

        self.assertTrue(
            approximately_evaluate_abst_for(
                property,
                scriptsizec.SCRIPTSIZE_C_GRAMMAR,
                gg.GrammarGraph.from_grammar(scriptsizec.SCRIPTSIZE_C_GRAMMAR),
                {language.Constant("start", "<start>"): ((), inp)},
                inp.trie().trie).is_true())

    def test_evaluation_xml_balance(self):
        property = parse_isla("""
forall <xml-tree> container="<{<id> opid}><inner-xml-tree></{<id> clid}>" in start:
  (= opid clid)""", xml_lang.XML_GRAMMAR)

        inp = language.DerivationTree.from_parse_tree(
            next(EarleyParser(xml_lang.XML_GRAMMAR).parse("<a>b</a>")))

        self.assertTrue(
            approximately_evaluate_abst_for(
                property,
                xml_lang.XML_GRAMMAR,
                gg.GrammarGraph.from_grammar(xml_lang.XML_GRAMMAR),
                {language.Constant("start", "<start>"): ((), inp)},
                inp.trie().trie).is_true())

    def test_icmp_ping_request(self):
        expected_checksum_constraint = parse_abstract_isla("""
forall <icmp_message> container in start:
  exists <checksum> checksum in container:
    internet_checksum(container, checksum)""", ICMP_GRAMMAR, semantic_predicates=ISLEARN_STANDARD_SEMANTIC_PREDICATES)

        # -> Disjunctive invariant!
        type_constraint = parse_abstract_isla("""
(forall <icmp_message> container in start:
   exists <type> elem in container:
     (= elem "00 ") or
forall <icmp_message> container_0 in start:
  exists <type> elem_0 in container_0:
    (= elem_0 "08 ")))""", ICMP_GRAMMAR)

        code_constraint = parse_abstract_isla("""
forall <icmp_message> container in start:
  exists <code> elem in container:
    (= elem "00 ")""", ICMP_GRAMMAR)

        inputs: Set[language.DerivationTree] = set([])
        for _ in range(50):
            size = random.randint(0, 16) * 2
            random_text = ''.join(random.choice("ABCDEF" + string.digits) for _ in range(size))
            payload = bytes(hex_to_bytes(random_text))

            icmp_packet = icmp.ICMP(
                icmp.Types.EchoReply if random.random() < .5 else icmp.Types.EchoRequest,
                payload=payload,
                identifier=random.randint(0, 0xFFFF),
                sequence_number=random.randint(0, 0xFFFF // 2)  # // 2 because of short format
            ).packet
            packet_bytes = list(bytearray(icmp_packet))
            icmp_packet_hex_dump = bytes_to_hex(packet_bytes).upper()

            inputs.add(language.DerivationTree.from_parse_tree(
                PEGParser(ICMP_GRAMMAR).parse(icmp_packet_hex_dump + " ")[0]))

        result = InvariantLearner(
            ICMP_GRAMMAR,
            prop=None,
            activated_patterns={
                "Checksums",
                "String Existence",
            },
            positive_examples=inputs,
            max_disjunction_size=2,
            filter_inputs_for_learning_by_kpaths=False,
            min_recall=1
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        self.assertIn(expected_checksum_constraint, result.keys())
        self.assertIn(code_constraint, result.keys())
        # self.assertTrue(any(
        #     implies(rf, type_constraint, ICMP_GRAMMAR) and
        #     implies(type_constraint, rf, ICMP_GRAMMAR) for rf in result.keys()))

    def test_ip_icmp_ping_request(self):
        import scapy.all as scapy
        ip_header_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <header_checksum> checksum in container:
    internet_checksum(container, checksum)""", IPv4_GRAMMAR)

        protocol_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <protocol> elem in container:
    (= elem "01 ")""", IPv4_GRAMMAR)

        identification_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <identification> elem in container:
    (= elem "00 01 ")""", IPv4_GRAMMAR)

        icmp_type_constraint = parse_abstract_isla("""
forall <data> container in start:
  exists <byte> elem in container:
    (nth("1", elem, container) and
    (= elem "08 "))""", IPv4_GRAMMAR)

        icmp_code_constraint = parse_abstract_isla("""
forall <data> container in start:
  exists <byte> elem in container:
    (nth("2", elem, container) and
    (= elem "00 "))""", IPv4_GRAMMAR)

        length_constraint = parse_abstract_isla("""
forall <ip_message> container in start:
  exists <total_length> length_field in container:
    exists int decimal:
      (hex_to_decimal(length_field, decimal) and
      (= (div (str.len (str.replace_all container " " "")) 2) (str.to.int decimal)))""", IPv4_GRAMMAR)

        inputs: Set[language.DerivationTree] = set([])
        for _ in range(30):
            size = random.randint(0, 16) * 2
            random_text = ''.join(random.choice("ABCDEF" + string.digits) for _ in range(size))
            payload = bytes(hex_to_bytes(random_text))

            icmp = scapy.ICMP()
            icmp.payload = scapy.Raw(payload)
            icmp.id = random.randint(0, 0xFFFF // 2)  # // 2 because of short format
            icmp.seq = random.randint(0, 0xFFFF // 2)

            p = scapy.IP(dst="8.8.8.8") / icmp
            ip_packet_hex_dump = bytes_to_hex(list(bytes(p)))

            inputs.add(language.DerivationTree.from_parse_tree(
                PEGParser(IPv4_GRAMMAR).parse(ip_packet_hex_dump + " ")[0]))

        result = InvariantLearner(
            IPv4_GRAMMAR,
            prop=None,
            activated_patterns={
                "Checksums",
                "String Existence",
                "Positioned String Existence (CSV)",
                "Existence Length Field (Hex)",
            },
            positive_examples=inputs,
            filter_inputs_for_learning_by_kpaths=False,
            min_recall=1.0,
        ).learn_invariants()

        # print(len(result))
        # print("\n".join(map(
        #     lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
        #     {f: p for f, p in result.items() if p > .0}.items())))

        self.assertIn(ip_header_constraint, result.keys())
        self.assertIn(protocol_constraint, result.keys())
        self.assertIn(identification_constraint, result.keys())
        self.assertIn(icmp_type_constraint, result.keys())
        self.assertIn(icmp_code_constraint, result.keys())
        self.assertIn(length_constraint, result.keys())

    # @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_learn_graphviz(self):
        urls = [
            "https://raw.githubusercontent.com/ecliptik/qmk_firmware-germ/56ea98a6e5451e102d943a539a6920eb9cba1919/users/dennytom/chording_engine/state_machine.dot",
            "https://raw.githubusercontent.com/Ranjith32/linux-socfpga/30f69d2abfa285ad9138d24d55b82bf4838f56c7/Documentation/blockdev/drbd/disk-states-8.dot",
            # Below one is graph, not digraph
            "https://raw.githubusercontent.com/nathanaelle/wireguard-topology/f0e42d240624ca0aa801d890c1a4d03d5901dbab/examples/3-networks/topology.dot"
        ]

        positive_trees = []
        for url in urls:
            with urllib.request.urlopen(url) as f:
                dot_code = (re.sub(r"(^|\n)\s*//.*?(\n|$)", "", f.read().decode('utf-8'))
                            .replace("\\n", "\n")
                            .replace("\r\n", "\n")
                            .strip())
            positive_trees.append(
                language.DerivationTree.from_parse_tree(list(PEGParser(DOT_GRAMMAR).parse(dot_code))[0]))

        # positive_inputs = [
        #     "digraph { a -> b }",
        #     "graph gg { c -- x }",
        #     "graph { a; y; a -- y }",
        #     "digraph { a; b; c; d; c -> d; a -> c; }",
        # ]
        # positive_trees = [
        #     language.DerivationTree.from_parse_tree(list(PEGParser(DOT_GRAMMAR).parse(inp))[0])
        #     for inp in positive_inputs]

        negative_inputs = [
            "digraph { a -- b }",
            "graph gg { a -> b }",
            "graph { a; b; a -> b }",
            "digraph { a; b; c; d; c -- d; a -> b; }",
        ]
        negative_trees = [
            language.DerivationTree.from_parse_tree(list(PEGParser(DOT_GRAMMAR).parse(inp))[0])
            for inp in negative_inputs]

        ##############
        # repo = patterns_from_file()
        # candidates = InvariantLearner(
        #     DOT_GRAMMAR,
        #     None,
        #     positive_examples=positive_trees,
        #     exclude_nonterminals={
        #         "<WS>", "<WSS>", "<MWSS>",
        #         "<esc_or_no_string_endings>", "<esc_or_no_string_ending>", "<no_string_ending>", "<LETTER_OR_DIGITS>",
        #         "<LETTER>", "<maybe_minus>", "<maybe_comma>", "<maybe_semi>"
        #     }
        # ).generate_candidates(
        #     repo["String Existence"],
        #     positive_trees
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##############

        def prop(tree: language.DerivationTree) -> bool:
            return render_dot(tree) is True

        # TODO: There's a hidden balance property, e.g., in labels:
        #       For each opening < in a String there has to be a closing >.

        result = InvariantLearner(
            DOT_GRAMMAR,
            prop=prop,
            activated_patterns={"String Existence"},
            positive_examples=positive_trees,
            # negative_examples=negative_trees,
            target_number_positive_samples=15,
            # target_number_positive_samples_for_learning=6,
            target_number_negative_samples=20,
            max_disjunction_size=2,
            include_negations_in_disjunctions=False,
            filter_inputs_for_learning_by_kpaths=False,
            max_conjunction_size=2,
            min_recall=1,
            min_specificity=.9,
            # reduce_all_inputs=True,
            reduce_inputs_for_learning=True,
            generate_new_learning_samples=False,
            # do_generate_more_inputs=False,
            # reduce_inputs_for_learning=True,
            exclude_nonterminals={
                "<WS>", "<WSS>", "<MWSS>",
                "<esc_or_no_string_endings>", "<esc_or_no_string_ending>", "<no_string_ending>", "<LETTER_OR_DIGITS>",
                "<LETTER>", "<maybe_minus>", "<maybe_comma>", "<maybe_semi>"
            }
        ).learn_invariants(ensure_unique_var_names=False)

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p[0] > .0}.items())))

        for f in result.keys():
            for inp in positive_trees:
                self.assertTrue(
                    evaluate(f, inp, DOT_GRAMMAR).is_true(),
                    f"Not true for '{inp}': {ISLaUnparser(f).unparse()}"
                )

        self.assertTrue(all(
            evaluate(f, inp, DOT_GRAMMAR).is_false()
            for f in result.keys()
            for inp in negative_trees
        ))

    def test_instantiate_nonterminal_placeholders_xml_tag(self):
        xml_doc = (
            '<view:view xmlns:view="0">'
            '<cm:content xs="1" xmlns:d="0" xmlns:cm="0" view:ce="tl">'
            '<cm:description> </cm:description>'
            '<cm:content>8</cm:content>'
            '</cm:content>'
            '</view:view>')
        tree = language.DerivationTree.from_parse_tree(
            list(EarleyParser(xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES).parse(xml_doc))[0])
        repo = patterns_from_file()
        learner = InvariantLearner(
            xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES,
            positive_examples={tree},
            # mexpr_expansion_limit=3,
            # max_nonterminals_in_mexpr=4,
            exclude_nonterminals={
                "<id-start-char>",
                "<id-chars>",
                "<id-char>",
                "<text-char>",
                "<text>",
                "<xml-open-tag>",
                "<xml-openclose-tag>",
                "<xml-close-tag>",
            }
        )

        instantiations = learner._instantiate_nonterminal_placeholders(
            next(iter(repo["Def-Use (XML-Tag Strict)"])),
            create_input_reachability_relation([tree]))

        expected = parse_abstract_isla("""
forall <xml-tree> xml_tree="{<?MATCHEXPR(<id-no-prefix> prefix_use)>}" in start:
  exists <xml-tree> outer_tag="{<?MATCHEXPR(<xml-attribute> cont_attribute)>}" in start:
    (inside(xml_tree, outer_tag) and
     exists <xml-attribute> def_attribute="{<?MATCHEXPR(<id-no-prefix> ns_prefix, <id-no-prefix> prefix_def)>}" 
         in cont_attribute:
       ((= ns_prefix <?STRING>) and
        (= prefix_use prefix_def)))
""", xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES)

        print(AbstractISLaUnparser(expected).unparse())
        print(len(instantiations))
        print("\n".join(map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations)))

        self.assertIn(
            AbstractISLaUnparser(expected).unparse(),
            map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations))

    def test_instantiate_mexpr_placeholders_xml(self):
        repo = patterns_from_file()
        learner = InvariantLearner(
            xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES,
            mexpr_expansion_limit=4,
            max_nonterminals_in_mexpr=4,
            exclude_nonterminals={
                "<id-start-char>",
                "<id-chars>",
                "<id-char>",
                "<text-char>",
                "<text>",
                "<xml-open-tag>",
                "<xml-openclose-tag>",
                "<xml-close-tag>",
            }
        )

        partial_inst = parse_abstract_isla("""
forall <xml-tree> xml_tree="{<?MATCHEXPR(<id-no-prefix> prefix_use)>}" in start:
  exists <xml-tree> outer_tag="{<?MATCHEXPR(<xml-attribute> cont_attribute)>}" in start:
    (inside(xml_tree, outer_tag) and
     exists <xml-attribute> def_attribute="{<?MATCHEXPR(<id-no-prefix> ns_prefix, <id-no-prefix> prefix_def)>}" 
         in cont_attribute:
       ((= ns_prefix <?STRING>) and
        (= prefix_use prefix_def)))
""", xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES)

        expected = parse_abstract_isla("""
forall <xml-tree> xml_tree="<{<id-no-prefix> prefix_use}:<id-no-prefix>><inner-xml-tree><xml-close-tag>" in start:
  exists <xml-tree> outer_tag="<<id> {<xml-attribute> cont_attribute}><inner-xml-tree><xml-close-tag>" in start:
    (inside(xml_tree, outer_tag) and
     exists <xml-attribute> def_attribute="{<id-no-prefix> ns_prefix}:{<id-no-prefix> prefix_def}=\\\"<text>\\\"" 
         in cont_attribute:
       ((= ns_prefix <?STRING>) and
        (= prefix_use prefix_def)))
""", xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES)

        instantiations = learner._instantiate_mexpr_placeholders({partial_inst})

        print(AbstractISLaUnparser(expected).unparse())
        print(len(instantiations))
        print("\n".join(map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations)))

        self.assertIn(
            AbstractISLaUnparser(expected).unparse(),
            map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations))

    def test_instantiate_nonterminal_placeholders_racket(self):
        racket_code = """
(define (point-origin-calc x y)
  (+ (+ (+ x x) (+ y y))))""".strip()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])

        repo = patterns_from_file()
        learner = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            None,
            positive_examples={tree},
            exclude_nonterminals={
                "<maybe_wss_names>",
                "<wss_exprs>",
                "<maybe_cond_args>",
                "<strings_mwss>",
                "<NAME_CHAR>",
                "<ONENINE>",
                "<ESC_OR_NO_STRING_ENDINGS>",
                "<ESC_OR_NO_STRING_ENDING>",
                "<NO_STRING_ENDING>",
                "<CHARACTER>",
                "<DIGIT>",
                "<LETTERORDIGIT>",
                "<MWSS>",
                "<WSS>",
                "<WS>",
                "<maybe_comments>",
                "<COMMENT>",
                "<HASHDIRECTIVE>",
                "<NOBR>",
                "<NOBRs>",
                "<test_case>",
                "<library_require>",
                "<pkg>",
                "<SYMBOL>",
                "<NUMBER>",
                "<DIGITS>",
                "<MAYBE_DIGITS>",
                "<INT>",
                "<BOOLEAN>",
                "<STRING>",
                "<program>",  # TODO: Remove for evaluation
                "<def_or_exprs>",  # TODO: Remove for evaluation
                "<def_or_expr>",  # TODO: Remove for evaluation
                "<cond_args>",  # TODO: Remove for evaluation
            },
        )

        expected = parse_abstract_isla("""
forall <expr> attribute="{<?MATCHEXPR(<name> prefix_use)>}" in start:
  ((= prefix_use <?STRING>) or
  exists <definition> outer_tag="{<?MATCHEXPR(<WSS_NAMES> cont_attribute)>}" in start:
    (inside(attribute, outer_tag) and
    exists <NAME> def_attribute="{<?MATCHEXPR(<NAME_CHARS> prefix_def)>}" in cont_attribute:
      (= prefix_use prefix_def)))""", RACKET_BSL_GRAMMAR)

        instantiations = learner._instantiate_nonterminal_placeholders(
            next(iter(repo["Def-Use (XML-Attr)"])),
            create_input_reachability_relation([tree]))

        print(AbstractISLaUnparser(expected).unparse())
        print(len(instantiations))
        print("\n".join(map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations)))

        self.assertIn(
            AbstractISLaUnparser(expected).unparse(),
            map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations))

    def test_infer_mexpr_racket(self):
        learner = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            None,
            positive_examples={},
        )

        mexprs = learner._infer_mexpr("<expr>", ("<name>",))
        self.assertTrue(mexprs)

        print(mexprs)

        self.assertTrue(
            any(tuple([re.sub(r'<([a-zA-Z_]+)--?[0-9]+>', r'<\1>', elem) for elem in mexpr_elements]) ==
                ('<maybe_comments>', '<MWSS>', '(', '<MWSS>', '<name>', '<wss_exprs>', '<MWSS>', ')')
                for mexpr_elements in mexprs))
        self.assertTrue(
            any(tuple([re.sub(r'<([a-zA-Z_]+)--?[0-9]+>', r'<\1>', elem) for elem in mexpr_elements]) ==
                ('<maybe_comments>', '<MWSS>', '<name>')
                for mexpr_elements in mexprs))

    def test_infer_mexpr_xml_tree(self):
        learner = InvariantLearner(
            xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES,
            None,
            positive_examples={},
            mexpr_expansion_limit=2,
            max_nonterminals_in_mexpr=4,  # Optional
        )

        mexprs = learner._infer_mexpr("<xml-tree>", ("<id>", "<inner-xml-tree>", "<id>"))
        self.assertTrue(mexprs)

        # print(mexprs)

        self.assertTrue(
            any(tuple([re.sub(r'<([a-zA-Z_-]+?)--?[0-9]+>', r'<\1>', elem) for elem in mexpr_elements]) ==
                ('<', '<id>', ' ', '<xml-attribute>', '>', '<inner-xml-tree>', '</', '<id>', '>')
                for mexpr_elements in mexprs))

        self.assertTrue(
            any(tuple([re.sub(r'<([a-zA-Z_-]+?)--?[0-9]+>', r'<\1>', elem) for elem in mexpr_elements]) ==
                ('<', '<id>', '>', '<inner-xml-tree>', '</', '<id>', '>')
                for mexpr_elements in mexprs))

    def test_infer_mexpr_xml_attr(self):
        learner = InvariantLearner(
            xml_lang.XML_GRAMMAR_WITH_NAMESPACE_PREFIXES,
            None,
            positive_examples={},
            mexpr_expansion_limit=3,
            max_nonterminals_in_mexpr=3,
        )

        mexprs = learner._infer_mexpr("<xml-attribute>", ("<id-no-prefix>", "<id-no-prefix>",))
        self.assertTrue(mexprs)

        print(mexprs)

        self.assertTrue(
            any(tuple([re.sub(r'<([a-zA-Z_-]+?)--?[0-9]+>', r'<\1>', elem) for elem in mexpr_elements]) ==
                ('<id-no-prefix>', ':', '<id-no-prefix>', '="', '<text>', '"')
                for mexpr_elements in mexprs))

    def test_instantiate_mexpr_placeholders_racket(self):
        learner = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            None,
            positive_examples={},
            mexpr_expansion_limit=1,
            max_nonterminals_in_mexpr=9,
            exclude_nonterminals={
                "<maybe_wss_names>",
                "<wss_exprs>",
                "<maybe_cond_args>",
                "<strings_mwss>",
                "<NAME_CHAR>",
                "<ONENINE>",
                "<ESC_OR_NO_STRING_ENDINGS>",
                "<ESC_OR_NO_STRING_ENDING>",
                "<NO_STRING_ENDING>",
                "<CHARACTER>",
                "<DIGIT>",
                "<LETTERORDIGIT>",
                "<MWSS>",
                "<WSS>",
                "<WS>",
                "<maybe_comments>",
                "<COMMENT>",
                "<HASHDIRECTIVE>",
                "<NOBR>",
                "<NOBRs>",
                "<test_case>",
                "<library_require>",
                "<pkg>",
                "<SYMBOL>",
                "<NUMBER>",
                "<DIGITS>",
                "<MAYBE_DIGITS>",
                "<INT>",
                "<BOOLEAN>",
                "<STRING>",
                "<program>",  # TODO: Remove for evaluation
                "<def_or_exprs>",  # TODO: Remove for evaluation
                "<def_or_expr>",  # TODO: Remove for evaluation
                "<cond_args>",  # TODO: Remove for evaluation
            },
        )

        partial_inst = parse_abstract_isla("""
forall <expr> attribute="{<?MATCHEXPR(<name> prefix_use)>}" in start:
  ((= prefix_use <?STRING>) or
  exists <definition> outer_tag="{<?MATCHEXPR(<WSS_NAMES> cont_attribute, <expr> contained_tree)>}" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<?MATCHEXPR(<NAME_CHARS> prefix_def)>}" in cont_attribute:
      (= prefix_use prefix_def)))""", RACKET_BSL_GRAMMAR)

        expected = parse_abstract_isla("""
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use <?STRING>) or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS>{<expr> contained_tree}<MWSS>)" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))""")

        instantiations = learner._instantiate_mexpr_placeholders({partial_inst})

        # print(AbstractISLaUnparser(expected).unparse())
        # print(len(instantiations))
        # print("\n".join(map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations)))

        self.assertIn(
            AbstractISLaUnparser(expected).unparse(),
            map(lambda f: AbstractISLaUnparser(f).unparse(), instantiations))

    def test_instantiate_string_placeholders_racket(self):
        property = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use <?STRING>) or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS>{<expr> contained_tree}<MWSS>)" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))"""

        expected = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use "+") or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS>{<expr> contained_tree}<MWSS>)" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))"""

        racket_code = """
(define (point-origin-calc x y)
  (+ (+ (+ x x) (+ y y))))""".strip()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])

        instantiations = InvariantLearner(
            RACKET_BSL_GRAMMAR,
        )._instantiate_string_placeholders(
            {parse_abstract_isla(property, RACKET_BSL_GRAMMAR)},
            [tree.trie().trie])

        print(len(instantiations))
        print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), instantiations)))

        self.assertIn(parse_abstract_isla(expected, RACKET_BSL_GRAMMAR), instantiations)

    def test_instantiate_dstrings_placeholders_racket(self):
        property = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use <?DSTRINGS>) or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS>{<expr> contained_tree}<MWSS>)" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))"""

        expected = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((((= prefix_use "*") or
      (= prefix_use "sqrt")) or
     (= prefix_use "+")) or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS>{<expr> contained_tree}<MWSS>)" in start:
    (inside(attribute, contained_tree) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))"""

        racket_code = """
(define (point-origin-calc x y)
  (sqrt (+ (* x x)
           (* y y))))""".strip()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])

        instantiations = InvariantLearner(
            RACKET_BSL_GRAMMAR,
        )._instantiate_string_placeholders(
            {parse_abstract_isla(property, RACKET_BSL_GRAMMAR)},
            [tree.trie().trie])

        print(len(instantiations))
        print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), instantiations)))

        self.assertTrue(any(
            equivalent(parse_abstract_isla(expected, RACKET_BSL_GRAMMAR), inst, RACKET_BSL_GRAMMAR)
            for inst in instantiations))

    def test_simplified_racket_xml_defuse_prop_get_candidates(self):
        racket_code = """
(define (point-origin-calc x y)
  (+ (+ (+ x x) (+ y y))))""".strip()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])

        # This is an instantiation of the XML predicate
        expected_defuse_property = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((= prefix_use "+") or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS><expr><MWSS>)" in start:
    (inside(attribute, outer_tag) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def)))"""

        repo = patterns_from_file()
        candidates = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            prop=None,
            positive_examples={tree},
            exclude_nonterminals={
                "<maybe_wss_names>",
                "<wss_exprs>",
                "<maybe_cond_args>",
                "<strings_mwss>",
                "<NAME_CHAR>",
                "<ONENINE>",
                "<ESC_OR_NO_STRING_ENDINGS>",
                "<ESC_OR_NO_STRING_ENDING>",
                "<NO_STRING_ENDING>",
                "<CHARACTER>",
                "<DIGIT>",
                "<LETTERORDIGIT>",
                "<MWSS>",
                "<WSS>",
                "<WS>",
                "<maybe_comments>",
                "<COMMENT>",
                "<HASHDIRECTIVE>",
                "<NOBR>",
                "<NOBRs>",
                "<test_case>",
                "<library_require>",
                "<pkg>",
                "<SYMBOL>",
                "<NUMBER>",
                "<DIGITS>",
                "<MAYBE_DIGITS>",
                "<INT>",
                "<BOOLEAN>",
                "<STRING>",
                "<program>",  # TODO: Remove for evaluation
                "<def_or_exprs>",  # TODO: Remove for evaluation
                "<def_or_expr>",  # TODO: Remove for evaluation
                "<cond_args>",  # TODO: Remove for evaluation
            },
        ).generate_candidates(
            repo["Def-Use (XML-Attr)"],
            {tree}
        )

        print(len(candidates))
        print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))

        self.assertIn(parse_abstract_isla(expected_defuse_property, RACKET_BSL_GRAMMAR), candidates)

    def test_racket_function_defuse_property_get_candidates(self):
        racket_code = """
#lang htdp/bsl
(define (f x) x)
(f 13)
(* 1 2)
(+ 1 2)
""".strip()

        positive_trees = [
            language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])]

        def prop(tree: language.DerivationTree) -> bool:
            return load_racket(tree) is True

        expected_defuse_property = """
forall <expr> use_ctx="<maybe_comments><MWSS>(<MWSS>{<name> use}<wss_exprs><MWSS>)" in start:
  ((= use "*") or (= use "+") or
    exists <definition> def_ctx="(<MWSS>define<MWSS>(<MWSS>{<name> def}<WSS_NAMES><MWSS>)<MWSS><expr><MWSS>)" in start:
      ((before(def_ctx, use_ctx) and
      (= use def))))
""".strip()

        self.assertTrue(evaluate(
            parse_isla(expected_defuse_property, structural_predicates={BEFORE_PREDICATE}),
            positive_trees[0],
            RACKET_BSL_GRAMMAR).is_true())

        repo = patterns_from_file()
        candidates = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            prop,
            mexpr_expansion_limit=1,
            max_nonterminals_in_mexpr=9,
            positive_examples={positive_trees[0]},
            exclude_nonterminals={
                "<maybe_wss_names>",
                "<wss_exprs>",
                "<maybe_cond_args>",
                "<strings_mwss>",
                "<NAME_CHAR>",
                "<ONENINE>",
                "<ESC_OR_NO_STRING_ENDINGS>",
                "<ESC_OR_NO_STRING_ENDING>",
                "<NO_STRING_ENDING>",
                "<CHARACTER>",
                "<DIGIT>",
                "<LETTERORDIGIT>",
                "<MWSS>",
                "<WSS>",
                "<WS>",
                "<maybe_comments>",
                "<COMMENT>",
                "<HASHDIRECTIVE>",
                "<NOBR>",
                "<NOBRs>",
                "<test_case>",
                "<library_require>",
                "<pkg>",
                "<SYMBOL>",
                "<NUMBER>",
                "<DIGITS>",
                "<MAYBE_DIGITS>",
                "<INT>",
                "<BOOLEAN>",
                "<STRING>",
                "<program>",  # TODO: Remove for evaluation
                "<def_or_exprs>",  # TODO: Remove for evaluation
                "<def_or_expr>",  # TODO: Remove for evaluation
                "<cond_args>",  # TODO: Remove for evaluation
            },
        ).generate_candidates(
            repo["Def-Use (reST Strict Reserved Names)"],
            {positive_trees[0]}
        )

        print(len(candidates))
        print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))

        self.assertTrue(any(
            equivalent(parse_abstract_isla(expected_defuse_property, RACKET_BSL_GRAMMAR), inst, RACKET_BSL_GRAMMAR)
            for inst in candidates))

    def test_racket_defuse_property_get_candidates(self):
        # The racket syntax check is really expensive; therefore, reduction cannot be used efficiently.
        # Also, the HTDP examples are pretty small already.
        # urls = [
        #     f"https://github.com/johnamata/compsci/raw/"
        #     f"cfb0e48c151da1d3463f3f0faca9f666af22ee16/htdp/exercises/{str(i).rjust(3, '0')}.rkt"
        #     for i in range(1, 30)
        # ]
        #
        # positive_trees = []
        # for url in urls:
        #     with urllib.request.urlopen(url) as f:
        #         racket_code = f.read().decode('utf-8').replace("\\n", "\n").replace("\r\n", "\n").strip()
        #     if "GRacket" in racket_code:  # Not a real racket file
        #         continue
        #     positive_trees.append(
        #         language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0]))

        def prop(tree: language.DerivationTree) -> bool:
            return load_racket(tree) is True

        racket_code = """
(define (point-origin-calc x y)
  (sqrt (+ (* x x)
           (* y y))))""".strip()

        positive_trees = [
            language.DerivationTree.from_parse_tree(list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])]

        # This is an instantiation of the XML predicate
        expected_defuse_property = """
forall <expr> attribute="<maybe_comments><MWSS>{<name> prefix_use}" in start:
  ((((= prefix_use "*") or
    (= prefix_use "+")) or
   (= prefix_use "sqrt")) or
  exists <definition> outer_tag="(<MWSS>define<MWSS>(<MWSS><name>{<WSS_NAMES> cont_attribute}<MWSS>)<MWSS><expr><MWSS>)" in start:
    (inside(attribute, outer_tag) and
    exists <NAME> def_attribute="{<NAME_CHARS> prefix_def}" in cont_attribute:
      (= prefix_use prefix_def))))""".strip()

        self.assertTrue(evaluate(
            parse_isla(expected_defuse_property, structural_predicates={IN_TREE_PREDICATE}),
            positive_trees[0],
            RACKET_BSL_GRAMMAR).is_true())

        repo = patterns_from_file()
        candidates = InvariantLearner(
            RACKET_BSL_GRAMMAR,
            prop,
            mexpr_expansion_limit=1,
            max_nonterminals_in_mexpr=9,
            positive_examples={positive_trees[0]},  # 8
            exclude_nonterminals={
                "<maybe_wss_names>",
                "<wss_exprs>",
                "<maybe_cond_args>",
                "<strings_mwss>",
                "<NAME_CHAR>",
                "<ONENINE>",
                "<ESC_OR_NO_STRING_ENDINGS>",
                "<ESC_OR_NO_STRING_ENDING>",
                "<NO_STRING_ENDING>",
                "<CHARACTER>",
                "<DIGIT>",
                "<LETTERORDIGIT>",
                "<MWSS>",
                "<WSS>",
                "<WS>",
                "<maybe_comments>",
                "<COMMENT>",
                "<HASHDIRECTIVE>",
                "<NOBR>",
                "<NOBRs>",
                "<test_case>",
                "<library_require>",
                "<pkg>",
                "<SYMBOL>",
                "<NUMBER>",
                "<DIGITS>",
                "<MAYBE_DIGITS>",
                "<INT>",
                "<BOOLEAN>",
                "<STRING>",
                "<program>",  # TODO: Remove for evaluation
                "<def_or_exprs>",  # TODO: Remove for evaluation
                "<def_or_expr>",  # TODO: Remove for evaluation
                "<cond_args>",  # TODO: Remove for evaluation
            },
        ).generate_candidates(
            repo["Def-Use (XML-Attr Disjunctive)"],
            {positive_trees[0]}
        )

        print(len(candidates))
        print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))

        self.assertTrue(any(
            equivalent(parse_abstract_isla(expected_defuse_property, RACKET_BSL_GRAMMAR), inst, RACKET_BSL_GRAMMAR)
            for inst in candidates))

    def test_load_patterns_from_file(self):
        patterns = patterns_from_file()
        self.assertTrue(patterns)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertIn("Def-Use", patterns)
        self.assertIn("Def-Use (C)", patterns)
        self.assertIn("Def-Use (XML-Attr)", patterns)
        self.assertNotIn("Def-Use (...)", patterns)


if __name__ == '__main__':
    unittest.main()
