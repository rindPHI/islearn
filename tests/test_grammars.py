import random
import re
import string
import unittest
import urllib.request

import scapy.all as scapy
from fuzzingbook.Parser import PEGParser
from isla import language
from pythonping import icmp

from languages import ICMP_GRAMMAR, IPv4_GRAMMAR, RACKET_BSL_GRAMMAR
from islearn.islearn_predicates import bytes_to_hex, hex_to_bytes


class TestGrammars(unittest.TestCase):
    def test_icmp(self):
        ping_request_message = """
08 00 83 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " "

        ping_response_message = """
00 00 8b 58 22 0e 00 00 62 1e fc f7 00 0b 08 75
08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17
18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27
28 29 2a 2b 2c 2d 2e 2f 30 31 32 33 34 35 36 37
""".upper().replace("\n", " ").strip() + " "

        parser = PEGParser(ICMP_GRAMMAR)
        self.assertTrue(parser.parse(ping_request_message))
        self.assertTrue(parser.parse(ping_response_message))

    def test_internet_checksum(self):
        num_1 = 0b1011
        num_2 = 0b0110
        length = max(num_1.bit_length(), num_2.bit_length())
        carry_mask = 2 ** length
        no_carry_mask = 2 ** length - 1
        sum = num_1 + num_2

        checksum = ((sum & no_carry_mask) + ((sum & carry_mask) >> length)) ^ no_carry_mask
        self.assertEqual(0b1101, checksum)

    def test_random_ping_packet_creation(self):
        size = random.randint(0, 32)
        random_text = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))

        icmp_obj = icmp.ICMP(
            icmp.Types.EchoRequest, payload=random_text, identifier=random.randint(0, 0xFFFF),
            sequence_number=random.randint(0, 0xFFFF))
        icmp_packet = icmp_obj.packet
        self.assertTrue(icmp_obj.is_valid)
        packet_bytes = list(bytearray(icmp_packet))

        icmp_packet_hex_dump = bytes_to_hex(packet_bytes)

        parser = PEGParser(ICMP_GRAMMAR)
        self.assertTrue(parser.parse(icmp_packet_hex_dump + " "))

    def test_ip_packet(self):
        for _ in range(10):
            size = random.randint(0, 16) * 2
            random_text = ''.join(random.choice("ABCDEF" + string.digits) for _ in range(size))
            payload = bytes(hex_to_bytes(random_text))

            icmp = scapy.ICMP()
            icmp.payload = scapy.Raw(payload)
            icmp.id = random.randint(0, 0xFFFF // 2)  # // 2 because of short format
            icmp.seq = random.randint(0, 0xFFFF // 2)

            p = scapy.IP(dst="8.8.8.8") / icmp
            ip_packet_hex_dump = bytes_to_hex(list(bytes(p)))

            self.assertTrue(PEGParser(IPv4_GRAMMAR).parse(ip_packet_hex_dump + " "))

    def test_racket_bsl_grammar(self):
        # Load all Racket BSL examples from a How to Design Programs exercise solutions repository.
        urls = [
            f"https://github.com/johnamata/compsci/raw/"
            f"cfb0e48c151da1d3463f3f0faca9f666af22ee16/htdp/exercises/{str(i).rjust(3, '0')}.rkt"
            for i in range(1, 30)
        ]

        for url in urls:
            with urllib.request.urlopen(url) as f:
                racket_code = f.read().decode('utf-8')
                if "GRacket" in racket_code:
                    # Not a real racket file
                    continue
                racket_code = racket_code.replace("\\n", "\n")
                racket_code = racket_code.replace("\r\n", "\n")
                racket_code = racket_code.strip()

                try:
                    tree = language.DerivationTree.from_parse_tree(
                        list(PEGParser(RACKET_BSL_GRAMMAR).parse(racket_code))[0])
                    if "(define" in racket_code:
                        self.assertTrue(
                            tree.filter(lambda t: t.value == "<definition>"),
                            f"<definition> node expected in URL {url}, program\n{racket_code}"
                        )
                except SyntaxError as e:
                    self.fail(f"Failed to parse URL {url} ({e}), file:\n\n{racket_code}")

    def test_racket_bsl_grammar_2(self):
        racket_code = """
#lang htdp/bsl

;;Define a function that consumes two numbers, x and y, and that computes the distance of point (x,y)
;; to the origin.

(define (point-origin-calc x y)
  (sqrt (+ (* x x)
           (* y y))))""".strip()

        parser = PEGParser(RACKET_BSL_GRAMMAR, log=True)

        tree = language.DerivationTree.from_parse_tree(list(parser.parse(racket_code))[0])
        self.assertTrue(tree.filter(lambda t: t.value == "<definition>"))


if __name__ == '__main__':
    unittest.main()
