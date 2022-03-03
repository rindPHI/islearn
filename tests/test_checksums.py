import unittest

from fuzzingbook.Parser import PEGParser
from isla import language
from isla.evaluator import evaluate
from isla.language import parse_isla

from grammars import ICMP_GRAMMAR
from islearn.checksums import compute_internet_checksum, internet_checksum, INTERNET_CHECKSUM_PREDICATE
from islearn.learner import approximately_evaluate_abst_for


class TestGrammars(unittest.TestCase):
    def test_compute_internet_checksum(self):
        self.assertEqual(0b1101, compute_internet_checksum([0b1011, 0b0110], length=4))

    def test_internet_checksum_predicate(self):
        parser = PEGParser(ICMP_GRAMMAR)

        ping_request_message = language.DerivationTree.from_parse_tree(parser.parse("""
08 00 83 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " ")[0])

        checksum_tree = ping_request_message.filter(lambda t: t.value == "<checksum>", enforce_unique=True)[0][1]
        self.assertTrue(internet_checksum(None, ping_request_message, checksum_tree).true())

        ping_response_message = language.DerivationTree.from_parse_tree(parser.parse("""
00 00 8b 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " ")[0])

        checksum_tree = ping_response_message.filter(lambda t: t.value == "<checksum>", enforce_unique=True)[0][1]
        self.assertTrue(True, internet_checksum(None, ping_response_message, checksum_tree).true())

        wrong_ping_request_message = language.DerivationTree.from_parse_tree(parser.parse("""
08 00 00 00 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " ")[0])

        checksum_tree = wrong_ping_request_message.filter(lambda t: t.value == "<checksum>", enforce_unique=True)[0][1]
        self.assertEquals(
            "83 58 ", str(internet_checksum(None, wrong_ping_request_message, checksum_tree).result[checksum_tree]))

    def test_checksum_formula(self):
        messages = [
            """
08 00 83 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37""",
            """
00 00 8b 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37"""
        ]

        trees = [
            language.DerivationTree.from_parse_tree(
                PEGParser(ICMP_GRAMMAR).parse(inp.upper().replace("\n", " ").strip() + " ")[0])
            for inp in messages]

        checksum_constraint = parse_isla("""
forall <start> container in start:
  forall <checksum> checksum in start:
    internet_checksum(container, checksum)""", ICMP_GRAMMAR, semantic_predicates={INTERNET_CHECKSUM_PREDICATE})

        self.assertTrue(evaluate(checksum_constraint, trees[0], ICMP_GRAMMAR).is_true())
        self.assertTrue(evaluate(checksum_constraint, trees[1], ICMP_GRAMMAR).is_true())

        wrong_ping_request_message = language.DerivationTree.from_parse_tree(PEGParser(ICMP_GRAMMAR).parse("""
08 00 00 00 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " ")[0])

        self.assertTrue(evaluate(checksum_constraint, wrong_ping_request_message, ICMP_GRAMMAR).is_unknown())

        # The approximate evaluator should return False instead of Unknown
        self.assertTrue(approximately_evaluate_abst_for(
            checksum_constraint,
            ICMP_GRAMMAR,
            {language.Constant("start", "<start>"): ((), wrong_ping_request_message)},
            dict(wrong_ping_request_message.paths())).is_false())


if __name__ == '__main__':
    unittest.main()
