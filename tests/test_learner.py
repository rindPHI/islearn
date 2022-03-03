import copy
import json
import math
import random
import re
import string
import unittest
from typing import cast, Tuple, Set

import pytest
import scapy.all as scapy
from fuzzingbook.Grammars import srange
from fuzzingbook.Parser import EarleyParser, PEGParser
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.helpers import strip_ws
from isla.isla_predicates import STANDARD_SEMANTIC_PREDICATES, BEFORE_PREDICATE
from isla.language import parse_isla, ISLaUnparser
from isla_formalizations import scriptsizec, csv, xml_lang, rest
from isla_formalizations.csv import CSV_HEADERBODY_GRAMMAR
from pythonping import icmp

from grammars import toml_grammar, JSON_GRAMMAR, ICMP_GRAMMAR, IPv4_GRAMMAR
from islearn.islearn_predicates import hex_to_bytes, bytes_to_hex
from islearn.language import parse_abstract_isla, NonterminalPlaceholderVariable, ISLEARN_STANDARD_SEMANTIC_PREDICATES
from islearn.learner import patterns_from_file, InvariantLearner, \
    create_input_reachability_relation, InVisitor, approximately_evaluate_abst_for


class TestLearner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        raw_inputs = [
            "{int c;c < 0;}",
            "{17 < 0;}",
        ]
        inputs = [
            language.DerivationTree.from_parse_tree(
                next(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(inp)))
            for inp in raw_inputs]

        #########
        # candidates = InvariantLearner(
        #     scriptsizec.SCRIPTSIZE_C_GRAMMAR,
        #     prop,
        #     activated_patterns={"Def-Use (C)"},
        #     positive_examples=inputs
        # ).generate_candidates(
        #     patterns_from_file()["Def-Use (C)"],
        #     inputs)
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##########

        result = InvariantLearner(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            prop,
            activated_patterns={"Def-Use (C)"},
            positive_examples=inputs,
            target_number_positive_samples=7,
            target_number_positive_samples_for_learning=4
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
            min_precision=.0,
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

        self.assertIn(
            correct_property.strip(),
            map(lambda f: ISLaUnparser(f).unparse(), result.keys()))

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_learn_invariants_mexpr_xml(self):
        correct_property = """
forall <xml-tree> container="<{<id> opid}><inner-xml-tree></{<id> clid}>" in start:
  (= opid clid)"""

        def prop(tree: language.DerivationTree) -> bool:
            return xml_lang.validate_xml(tree) is True

        raw_inputs = [
            "<a>asdf</a>",
            "<b>xyz<c/><x>X</x></b>",
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
            positive_examples=inputs
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
    count(elem, "<raw-field>", num))"""

        def prop(tree: language.DerivationTree) -> bool:
            return evaluate(correct_property, tree, csv.CSV_GRAMMAR).is_true()

        ##################
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

        perfect_precision_formulas = [f for f, p in result.items() if p == 1]
        self.assertEqual(2, len(perfect_precision_formulas))
        self.assertIn(correct_property.strip(), [ISLaUnparser(f).unparse() for f in perfect_precision_formulas])

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_string_existence(self):
        correct_property = """
forall <json> container in start:
  exists <string> elem in container:
    (= elem \"""key""\")
"""

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
            list(map(lambda f: ISLaUnparser(f).unparse(), [r for r, p in result.items() if p > .0])))

    @pytest.mark.flaky(reruns=3, reruns_delay=2)
    def test_alhazen_sqrt_example(self):
        correct_property_1 = """
(forall <arith_expr> container in start:
   exists <maybe_minus> elem in container:
     (= elem "-") and
forall <arith_expr> container_0 in start:
  exists <function> elem_0 in container_0:
    (= elem_0 "sqrt"))"""

        correct_property_2_re = re.escape("""
(forall <arith_expr> container in start:
   exists <function> elem in container:
     (= elem "sqrt") and
forall <arith_expr> container_0 in start:
  exists <number> elem_0 in container_0:
    (<= (str.to.int elem_0) (str.to.int "-""".strip()) + r"[1-9][0-9]*" + re.escape('")))')

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

        self.assertTrue(all(evaluate(correct_property_1, tree, grammar) for tree in trees))

        result = InvariantLearner(
            grammar,
            prop,
            activated_patterns={"Existence Numeric String Smaller Than", "String Existence"},
            # activated_patterns={"String Existence"},
            positive_examples=trees
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p > .0}.items())))

        nonzero_precision_results = list(map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p > .0]))

        self.assertIn(correct_property_1.strip(), nonzero_precision_results)

        self.assertTrue(any(re.match(correct_property_2_re, r) for r in nonzero_precision_results))

    def test_learn_from_islearn_patterns_file(self):
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
        #     repo["String Existence"],  # repo["Universal"], # repo["Existence Strings Relative Order"]
        #     trees
        # )
        #
        # print(len(candidates))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), candidates)))
        #
        # return
        ##############

        result = InvariantLearner(
            toml_grammar,
            prop=None,
            activated_patterns={"String Existence", "Universal"},
            positive_examples=trees
        ).learn_invariants()

        print(len(result))
        # print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p > .0}.items())))

        nonzero_precision_results = list(map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p > .0]))

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

        correct_property_4 = """
forall <key> elem in start:
  (<= (str.len elem) (str.to.int "11"))"""

        self.assertIn(correct_property_1.strip(), nonzero_precision_results)
        self.assertIn(correct_property_2.strip(), nonzero_precision_results)
        self.assertIn(correct_property_3.strip(), nonzero_precision_results)
        self.assertIn(correct_property_4.strip(), nonzero_precision_results)

    def test_str_len_ph_instantiation(self):
        repo = patterns_from_file()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(toml_grammar).parse(str(repo)))[0])

        pattern = parse_abstract_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int <?STRING>))""")

        result = InvariantLearner(
            toml_grammar,
            prop=None,
        )._instantiate_string_placeholders(
            {pattern}, [dict(tree.paths())]
        )

        expected = language.parse_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int "10"))""", toml_grammar)

        # print(len(result))
        # print("\n".join(map(lambda candidate: ISLaUnparser(candidate).unparse(), result)))
        self.assertIn(expected, result)

    def test_str_len_ph_instantiations(self):
        repo = patterns_from_file()
        tree = language.DerivationTree.from_parse_tree(list(PEGParser(toml_grammar).parse(str(repo)))[0])

        pattern = parse_abstract_isla("""
forall <key> elem in start:
  (<= (str.len elem) (str.to.int <?STRING>))""")

        result = InvariantLearner(
            toml_grammar,
            prop=None,
        )._get_string_placeholder_instantiations(
            {pattern}, [dict(tree.paths())]
        )

        self.assertEqual(1, len(result))
        self.assertEqual(1, len(list(result.values())[0]))
        insts: Set[str] = list(list(result.values())[0].values())[0]

        self.assertIn("name", insts)
        self.assertIn("constraint", insts)
        self.assertIn("11", insts)
        self.assertIn("Types", insts)
        self.assertIn("Def-Use", insts)
        self.assertIn("Existential", insts)
        self.assertIn("Misc", insts)

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

        expected_constraint_2 = '''
forall <key_value> container="{<key> key} = {<value> value}" in start:
  ((not (= key "name")) or
  (str.in_re 
    value 
    (re.++ 
      (re.++ 
        (str.to_re """")
        (re.* (re.comp (re.union (str.to_re "\\n") (str.to_re """"))))) 
      (str.to_re """"))))'''

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
            {f: p for f, p in result.items() if p > .0}.items())))

        nonzero_precision_results = list(map(strip_ws, map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p > .0])))

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
            {f: p for f, p in result.items() if p > .0}.items())))

        nonzero_precision_results = list(map(strip_ws, map(
            lambda f: ISLaUnparser(f).unparse(),
            [r for r, p in result.items() if p > .0])))

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

        subtrees = tuple(inp.paths())

        self.assertTrue(approximately_evaluate_abst_for(
            property,
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            {language.Constant("start", "<start>"): ((), inp)},
            dict(subtrees)).is_true())

    def test_evaluation_xml_balance(self):
        property = parse_isla("""
forall <xml-tree> container="<{<id> opid}><inner-xml-tree></{<id> clid}>" in start:
  (= opid clid)""", xml_lang.XML_GRAMMAR)

        inp = language.DerivationTree.from_parse_tree(
            next(EarleyParser(xml_lang.XML_GRAMMAR).parse("<a>b</a>")))

        subtrees = tuple(inp.paths())

        self.assertTrue(approximately_evaluate_abst_for(
            property,
            xml_lang.XML_GRAMMAR,
            {language.Constant("start", "<start>"): ((), inp)},
            dict(subtrees)).is_true())

    def test_icmp_ping_request(self):
        expected_checksum_constraint = parse_abstract_isla("""
forall <icmp_message> container in start:
  exists <checksum> checksum in container:
    internet_checksum(container, checksum)""", ICMP_GRAMMAR, semantic_predicates=ISLEARN_STANDARD_SEMANTIC_PREDICATES)

        type_constraint = parse_abstract_isla("""
forall <icmp_message> container in start:
  exists <byte> elem in container:
    (nth("1", elem, container) and
    (= elem "08 "))""", ICMP_GRAMMAR)

        code_constraint = parse_abstract_isla("""
forall <icmp_message> container in start:
  exists <byte> elem in container:
    (nth("2", elem, container) and
    (= elem "00 "))""", ICMP_GRAMMAR)

        inputs: Set[language.DerivationTree] = set([])
        for _ in range(50):
            size = random.randint(0, 16) * 2
            random_text = ''.join(random.choice("ABCDEF" + string.digits) for _ in range(size))
            payload = bytes(hex_to_bytes(random_text))

            icmp_packet = icmp.ICMP(
                icmp.Types.EchoRequest,
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
                "Positioned String Existence (CSV)",
            },
            positive_examples=inputs
        ).learn_invariants()

        print(len(result))
        print("\n".join(map(
            lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(),
            {f: p for f, p in result.items() if p > .0}.items())))

        self.assertIn(expected_checksum_constraint, result.keys())
        self.assertIn(type_constraint, result.keys())
        self.assertIn(code_constraint, result.keys())

    def test_ip_icmp_ping_request(self):
        ip_header_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <header_checksum> checksum in container:
    internet_checksum(container, checksum)""", IPv4_GRAMMAR)

        protocol_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <protocol> elem in container:
    (nth("1", elem, container) and
    (= elem "01 "))""", IPv4_GRAMMAR)

        identification_constraint = parse_abstract_isla("""
forall <header> container in start:
  exists <identification> elem in container:
    (nth("1", elem, container) and
    (= elem "00 01 "))""", IPv4_GRAMMAR)

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
                "Positioned String Existence (CSV)",
                "Existence Length Field (Hex)",
            },
            positive_examples=inputs
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

    def test_load_patterns_from_file(self):
        patterns = patterns_from_file()
        self.assertTrue(patterns)
        self.assertGreaterEqual(len(patterns), 2)
        self.assertIn("Def-Use", patterns)
        self.assertIn("Def-Use (C)", patterns)
        self.assertIn("Def-Use (XML)", patterns)
        self.assertNotIn("Def-Use (...)", patterns)


if __name__ == '__main__':
    unittest.main()
