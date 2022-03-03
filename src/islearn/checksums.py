import re
import string
from typing import List, Sequence, Optional

from fuzzingbook.Grammars import srange
from fuzzingbook.Parser import PEGParser
from isla import language
from isla.type_defs import Grammar


def internet_checksum(
        _: Optional[Grammar],
        header: language.DerivationTree,
        checksum_tree: language.DerivationTree) -> language.SemPredEvalResult:
    if not header.is_complete():
        return language.SemPredEvalResult(None)

    checksum_grammar = {
        "<start>": ["<checksum>"],
        "<checksum>": ["<byte><byte>"],
        "<byte>": ["<zerof><zerof> "],
        "<zerof>": srange(string.digits + "ABCDEF")
    }

    checksum_tree_str = re.sub(r"\s+", "", str(checksum_tree))
    if not len(checksum_tree_str) % 2 == 0:
        return language.SemPredEvalResult(False)

    zeroes = "".join("0" for _ in range(len(checksum_tree_str)))
    if str(checksum_tree).endswith(" "):
        zeroes += " "

    zero_checksum = ("<checksum>", [(zeroes, [])])

    header_wo_checksum = header.replace_path(
        header.find_node(checksum_tree),
        language.DerivationTree.from_parse_tree(zero_checksum))

    header_bytes: List[int] = hex_to_bytes(str(header_wo_checksum))

    checksum_value = int_to_hex(compute_internet_checksum(header_bytes)).upper() + " "
    if len(checksum_value) < 6:
        assert len(checksum_value) == 3
        checksum_value = "00 " + checksum_value

    checksum_value_nospace = re.sub(r"\s+", "", str(checksum_value))
    if checksum_value_nospace == checksum_tree_str:
        return language.SemPredEvalResult(True)

    checksum_parser = PEGParser(checksum_grammar, start_symbol="<checksum>")
    new_checksum_tree = language.DerivationTree.from_parse_tree(
        list(checksum_parser.parse(checksum_value))[0])

    if str(new_checksum_tree) == str(checksum_tree):
        return language.SemPredEvalResult(True)

    return language.SemPredEvalResult({checksum_tree: new_checksum_tree})


INTERNET_CHECKSUM_PREDICATE = language.SemanticPredicate("internet_checksum", 2, internet_checksum, binds_tree=False)


def compute_internet_checksum(inp_bytes: Sequence[int], length=16) -> int:
    if len(inp_bytes) % 2 != 0:
        inp_bytes = tuple(inp_bytes) + (0x00,)

    assert all(inp_byte <= 0xFF for inp_byte in inp_bytes)
    assert length < 8 or length % 8 == 0

    if length < 8:
        # This is only for testing purposes
        elements = list(inp_bytes)
    else:
        assert length == 16
        elements: List[int] = []
        for i in range(0, len(inp_bytes), 2):
            elements.append((inp_bytes[i] << 8) + inp_bytes[i + 1])

    carry_mask = 2 ** length
    no_carry_mask = 2 ** length - 1

    elements_sum = 0
    for element in elements:
        elements_sum += element
        elements_sum = ((elements_sum & no_carry_mask) + ((elements_sum & carry_mask) >> length))

    return elements_sum ^ no_carry_mask


def hex_to_bytes(hex_str: str) -> List[int]:
    return list(bytearray.fromhex(hex_str))


def int_to_hex(number: int, add_spaces=True) -> str:
    result = list(hex(number)[2:])
    if len(result) % 2 != 0:
        result.insert(0, "0")
    if add_spaces:
        for i in reversed(range(len(result))):
            if i > 0 and i % 2 == 0:
                result.insert(i, " ")
    return "".join(result)


def bytes_to_hex(inp_bytes: Sequence[int], add_spaces=True) -> str:
    result = list(bytearray(inp_bytes).hex())
    if add_spaces:
        for i in reversed(range(len(result))):
            if i > 0 and i % 2 == 0:
                result.insert(i, " ")
    return "".join(result)
