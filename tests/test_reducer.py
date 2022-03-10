import re
import unittest
import urllib.request

from fuzzingbook.Parser import PEGParser, EarleyParser
from isla.language import DerivationTree
from isla_formalizations import scriptsizec

from islearn.reducer import InputReducer
from languages import DOT_GRAMMAR, render_dot


class TestReducer(unittest.TestCase):
    def test_reduce_dot(self):
        dot_url = "https://raw.githubusercontent.com/ecliptik/qmk_firmware-germ/" \
                  "56ea98a6e5451e102d943a539a6920eb9cba1919/users/dennytom/chording_engine/state_machine.dot"

        with urllib.request.urlopen(dot_url) as f:
            dot_code = (re.sub(r"(^|\n)\s*//.*?(\n|$)", "", f.read().decode('utf-8'))
                        .replace("\\n", "\n")
                        .replace("\r\n", "\n")
                        .strip())
        tree = DerivationTree.from_parse_tree(list(PEGParser(DOT_GRAMMAR).parse(dot_code))[0])

        reducer = InputReducer(DOT_GRAMMAR, lambda t: render_dot(t) is True)

        result = reducer.reduce_by_smallest_subtree_replacement(tree)

        # print(tree)
        # print(result)

        self.assertTrue(render_dot(result) is True)
        self.assertLess(len(str(result)), len(str(tree)) / 3)

    def test_reduce_scriptsize_c(self):
        # This computes the GCD of m and n, result is in n after computation.
        c_code = """
{int n;int m;
if(n < 1) n = 0 - n;
if(m < 1) m = 0 - m;
int f = 0;
if(n < m) f = 1;
if(m < n) f = 1;
while(f) {
if(m < n) n = n - m;
if(n < m) m = m - n; else f = 0;}}
""".strip().replace("\n", "")

        tree = DerivationTree.from_parse_tree(list(EarleyParser(scriptsizec.SCRIPTSIZE_C_GRAMMAR).parse(c_code))[0])
        self.assertTrue(scriptsizec.compile_scriptsizec_clang(tree) is True)

        reducer = InputReducer(
            scriptsizec.SCRIPTSIZE_C_GRAMMAR,
            lambda t: scriptsizec.compile_scriptsizec_clang(t) is True,
            k=3
        )

        result = reducer.reduce_by_smallest_subtree_replacement(tree)

        # print(tree)
        # print(result)

        self.assertTrue(scriptsizec.compile_scriptsizec_clang(result) is True)
        self.assertLess(len(str(result)), len(str(tree)) / 2)


if __name__ == '__main__':
    unittest.main()
