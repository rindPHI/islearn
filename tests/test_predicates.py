import random
import string
import unittest

from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.language import parse_isla
from isla.parser import PEGParser
from pythonping import icmp

from islearn.islearn_predicates import compute_internet_checksum, internet_checksum, INTERNET_CHECKSUM_PREDICATE, \
    bytes_to_hex, \
    hex_to_bytes, hex_to_dec, hex_to_int
from islearn.learner import approximately_evaluate_abst_for
from islearn_example_languages import ICMP_GRAMMAR


class TestGrammars(unittest.TestCase):
    def test_compute_internet_checksum(self):
        self.assertEqual(0b1101, compute_internet_checksum((0b1011, 0b0110), length=4))

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

        # The approximate evaluator returns True, since the formula is satisfiable if
        # the assignments are changed.
        self.assertTrue(
            approximately_evaluate_abst_for(
                checksum_constraint,
                ICMP_GRAMMAR,
                gg.GrammarGraph.from_grammar(ICMP_GRAMMAR),
                {language.Constant("start", "<start>"): ((), wrong_ping_request_message)},
                wrong_ping_request_message.trie().trie).is_true())

    def test_checksum_random_ping_packet(self):
        checksum_constraint = parse_isla("""
forall <start> container in start:
  forall <checksum> checksum in start:
    internet_checksum(container, checksum)""", ICMP_GRAMMAR, semantic_predicates={INTERNET_CHECKSUM_PREDICATE})

        for _ in range(100):
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

            tree = language.DerivationTree.from_parse_tree(PEGParser(ICMP_GRAMMAR).parse(icmp_packet_hex_dump + " ")[0])

            self.assertTrue(evaluate(checksum_constraint, tree, ICMP_GRAMMAR).is_true())

    def test_hex_to_decimal(self):
        hex_str = "12 34 56 78 "
        parser = PEGParser(ICMP_GRAMMAR, start_symbol="<bytes>")
        hex_tree = language.DerivationTree.from_parse_tree(parser.parse(hex_str)[0])
        decimal_constant = language.Constant("decimal", "<decimal>")
        result = hex_to_dec(None, hex_tree, decimal_constant)
        self.assertEqual(str(hex_to_int(hex_str)), str(result.result[decimal_constant]))


if __name__ == '__main__':
    unittest.main()
