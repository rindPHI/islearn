import logging
import random
import string
import struct

from fuzzingbook.Parser import PEGParser
from grammar_graph import gg
from isla import language
from isla.evaluator import evaluate
from isla.language import ISLaUnparser
from pythonping import icmp

from islearn.islearn_predicates import hex_to_bytes, bytes_to_hex
from islearn.learner import InvariantLearner
from islearn_example_languages import ICMP_GRAMMAR

logging.basicConfig(level=logging.DEBUG)

random = random.SystemRandom()


def validate_icmp_ping(inp: language.DerivationTree | str | bytes) -> bool | str:
    if isinstance(inp, bytes):
        bytes_input = inp
    else:
        bytes_input = bytearray.fromhex(str(inp))

    if len(bytes_input) < 8:
        return f"Packet smaller than 8 bytes, actual length: {len(bytes_input)}"

    packet = icmp.ICMP()

    packet.raw = bytes_input

    (packet.message_type,
     packet.message_code,
     packet.received_checksum,
     packet.id,
     packet.sequence_number) = struct.unpack("bbHHh", bytes_input[:8])
    packet.payload = bytes_input[8:]

    if packet.message_type not in [icmp.Types.EchoRequest.type_id, icmp.Types.EchoReply.type_id]:
        return "Not a PING packet"

    if packet.message_code != 0:
        return "Invalid message code for PING"

    if not packet.is_valid:
        return "Invalid Checksum"

    return True


def create_random_valid_icmp_ping_packet_inp():
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

    return language.DerivationTree.from_parse_tree(
        PEGParser(ICMP_GRAMMAR).parse(icmp_packet_hex_dump + " ")[0])


def create_random_icmp_packet_inp():
    # This generates a valid ICMP packet (random type) with 90% probability;
    # with 20% probability, it creates a random (probably wrong) checksum.
    size = random.randint(0, 16) * 2
    random_text = ''.join(random.choice("ABCDEF" + string.digits) for _ in range(size))

    payload = bytes(hex_to_bytes(random_text))
    message_type = random.randint(0, 43)

    max_code = {key: 0 for key in range(0, 44)}
    max_code.update({
        3: 15,
        5: 3,
        12: 2,
        43: 4
    })

    icmp_obj = icmp.ICMP(
        icmp.Types.BadIPHeader,  # Cannot pass an int here due to a pythonping bug
        payload=payload,
        identifier=random.randint(0, 0xFFFF),
        sequence_number=random.randint(0, 0xFFFF // 2)  # // 2 because of (signed) short format
    )

    icmp_obj.message_type = message_type
    icmp_obj.message_code = random.randint(0, max_code[message_type])

    packet_bytes = list(bytearray(icmp_obj.packet))
    if random.random() < .4:
        packet_bytes[2] = random.randint(0x0, 0xF)
        packet_bytes[3] = random.randint(0x0, 0xF)

    icmp_packet_hex_dump = bytes_to_hex(packet_bytes).upper()

    return language.DerivationTree.from_parse_tree(
        PEGParser(ICMP_GRAMMAR).parse(icmp_packet_hex_dump + " ")[0])


if __name__ == "__main__":
    parser = PEGParser(ICMP_GRAMMAR)
    graph = gg.GrammarGraph.from_grammar(ICMP_GRAMMAR)

    positive_trees = [create_random_valid_icmp_ping_packet_inp() for _ in range(100)]

    negative_trees = []
    while len(negative_trees) < 100:
        inp = create_random_icmp_packet_inp()
        if validate_icmp_ping(inp) is not True:
            negative_trees.append(inp)

    random.shuffle(positive_trees)
    random.shuffle(negative_trees)

    learning_inputs = positive_trees[:50]
    negative_learning_inputs = negative_trees[:50]

    positive_validation_inputs = positive_trees[50:]
    negative_validation_inputs = negative_trees[50:]

    # Learn invariants
    result = InvariantLearner(
        ICMP_GRAMMAR,
        prop=None,
        activated_patterns={
            # "Checksums",
            "String Existence",
        },
        positive_examples=learning_inputs,
        negative_examples=negative_learning_inputs,
        max_disjunction_size=2,
        max_conjunction_size=2,
        filter_inputs_for_learning_by_kpaths=False,
        do_generate_more_inputs=False,
        min_recall=1,
        min_specificity=.7,
    ).learn_invariants()

    print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

    best_invariant, (specificity, sensitivity) = next(iter(result.items()))
    print(f"Best invariant (*estimated* specificity {specificity:.2f}, sensitivity {sensitivity:.2f}):")
    print(ISLaUnparser(best_invariant).unparse())

    # Finally, obtain confusion matrix entries.
    tp, tn, fp, fn = 0, 0, 0, 0

    for inp in positive_validation_inputs:
        if evaluate(best_invariant, inp, ICMP_GRAMMAR, graph=graph).is_true():
            tp += 1
        else:
            fn += 1

    for inp in negative_validation_inputs:
        if evaluate(best_invariant, inp, ICMP_GRAMMAR, graph=graph).is_true():
            fp += 1
        else:
            tn += 1

    print(f"TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}")
