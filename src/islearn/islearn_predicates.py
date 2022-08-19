import string
from functools import lru_cache
from typing import List, Sequence, Optional, Tuple

from isla import language
from isla.helpers import srange
from isla.parser import PEGParser
from isla.type_defs import Grammar

from islearn.helpers import remove_spaces


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

    checksum_tree_str = remove_spaces(checksum_tree)
    if not len(checksum_tree_str) % 2 == 0:
        return language.SemPredEvalResult(False)

    zeroes = "".join("0" for _ in range(len(checksum_tree_str)))
    if str(checksum_tree).endswith(" "):
        zeroes += " "

    zero_checksum = ("<checksum>", [(zeroes, [])])

    header_wo_checksum = header.replace_path(
        header.find_node(checksum_tree),
        language.DerivationTree.from_parse_tree(zero_checksum))

    header_bytes: Tuple[int] = tuple(hex_to_bytes(str(header_wo_checksum)))

    checksum_value = int_to_hex(compute_internet_checksum(header_bytes)).upper() + " "
    if len(checksum_value) < 6:
        assert len(checksum_value) == 3
        checksum_value = "00 " + checksum_value

    checksum_value_nospace = remove_spaces(checksum_value)
    if checksum_value_nospace == checksum_tree_str:
        return language.SemPredEvalResult(True)

    checksum_parser = PEGParser(checksum_grammar, start_symbol="<checksum>")
    new_checksum_tree = language.DerivationTree.from_parse_tree(
        list(checksum_parser.parse(checksum_value))[0])

    if str(new_checksum_tree) == str(checksum_tree):
        return language.SemPredEvalResult(True)

    return language.SemPredEvalResult({checksum_tree: new_checksum_tree})


INTERNET_CHECKSUM_PREDICATE = language.SemanticPredicate("internet_checksum", 2, internet_checksum, binds_tree=False)


@lru_cache(maxsize=128)
def compute_internet_checksum(inp_bytes: Tuple[int], length=16) -> int:
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


def hex_to_int(hex_str: str) -> int:
    length = len(remove_spaces(hex_str))
    if length % 2 != 0:
        hex_str = hex_str.rjust(length + 1, "0")
    return int.from_bytes(bytearray.fromhex(hex_str), byteorder="big")


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
    return "".join(result).upper()


def hex_to_dec(
        _: Optional[Grammar],
        hexadecimal: language.Variable | language.DerivationTree,
        decimal: language.Variable | language.DerivationTree) -> language.SemPredEvalResult:
    assert not isinstance(hexadecimal, language.Variable) or not isinstance(decimal, language.Variable)

    if (isinstance(hexadecimal, language.DerivationTree) and
            isinstance(decimal, language.DerivationTree) and
            not hexadecimal.is_complete() and
            not decimal.is_complete()):
        return language.SemPredEvalResult(None)

    bytes_grammar = {
        "<start>": ["<bytes>"],
        "<bytes>": ["<byte><bytes>", ""],
        "<byte>": ["<zerof><zerof> "],
        "<zerof>": srange(string.digits + "ABCDEF")
    }
    bytes_parser = PEGParser(bytes_grammar, start_symbol="bytes")

    if isinstance(hexadecimal, language.DerivationTree) and isinstance(decimal, language.DerivationTree):
        decimal_number = None
        hexadecimal_number = None
        if hexadecimal.is_complete():
            hexadecimal_number = hex_to_int(str(hexadecimal))
        if decimal.is_complete():
            decimal_number = int(str(decimal))

        if decimal_number == hexadecimal_number:
            return language.SemPredEvalResult(True)

        if hexadecimal.is_complete():
            return language.SemPredEvalResult({
                decimal: language.DerivationTree(
                    decimal.value,
                    (language.DerivationTree(str(hexadecimal_number), ()),))})
        else:
            return language.SemPredEvalResult({
                hexadecimal: language.DerivationTree(
                    hexadecimal.value,
                    (bytes_parser.parse(int_to_hex(decimal_number))[0]), )})

    if isinstance(hexadecimal, language.Variable) and isinstance(decimal, language.DerivationTree):
        if not decimal.is_complete():
            return language.SemPredEvalResult(None)

        return language.SemPredEvalResult({
            hexadecimal: language.DerivationTree(
                "<bytes>",
                (language.DerivationTree(
                    int_to_hex(int(str(decimal))),
                    ()),)
            )})

    if isinstance(decimal, language.Variable) and isinstance(hexadecimal, language.DerivationTree):
        if not hexadecimal.is_complete():
            return language.SemPredEvalResult(None)

        decimal_str = str(hex_to_int(str(hexadecimal)))

        return language.SemPredEvalResult({
            decimal: language.DerivationTree(
                "<decimal>",
                (language.DerivationTree(
                    decimal_str,
                    ()),)
            )})

    assert False


HEX_TO_DEC_PREDICATE = language.SemanticPredicate("hex_to_decimal", 2, hex_to_dec, binds_tree=False)
