import logging
import os.path
import re
import string
import urllib.request
import xml.etree.ElementTree as ET
from html import escape

import dill as pickle
from fuzzingbook.Grammars import srange
from fuzzingbook.Parser import PEGParser
from isla import language
from isla.language import ISLaUnparser
from isla_formalizations import xml_lang

from islearn.learner import InvariantLearner
from islearn.reducer import InputReducer

logging.basicConfig(level=logging.DEBUG)

XML_GRAMMAR_WITH_NAMESPACE_PREFIXES = {
    '<start>': ['<xml-tree>'],
    '<xml-tree>': ['<xml-open-tag><inner-xml-tree><xml-close-tag>', '<xml-openclose-tag>'],
    '<inner-xml-tree>': ['<xml-tree><inner-xml-tree>', '<xml-tree>', '<text>'],
    '<xml-open-tag>': ['<<id> <xml-attribute>>', '<<id>>'],
    '<xml-openclose-tag>': ['<<id> <xml-attribute>/>', '<<id>/>'],
    '<xml-close-tag>': ['</<id>>'],
    '<xml-attribute>': ['<id>="<text>" <xml-attribute>', '<id>="<text>"'],
    '<id>': ['<id-with-prefix>', '<id-no-prefix>'],
    "<id-start-char>": srange("_" + string.ascii_letters),
    "<id-chars>": ["<id-char><id-chars>", "<id-char>"],
    "<id-char>": ["<id-start-char>"] + srange("-." + string.digits),
    "<text>": ["<text-char><text>", ""],
    "<text-char>": [escape(c) for c in srange(string.ascii_letters + string.digits + "\"'. \t/?-,=:+_*${}%@|")],
    '<id-no-prefix>': ['<id-start-char><id-chars>', '<id-start-char>'],
    '<id-with-prefix>': ['<id-no-prefix>:<id-no-prefix>']
}


def prop(tree: language.DerivationTree) -> bool:
    return xml_lang.validate_xml(tree) is True


dirname = os.path.abspath(os.path.dirname(__file__))
parser = PEGParser(XML_GRAMMAR_WITH_NAMESPACE_PREFIXES)
reducer = InputReducer(XML_GRAMMAR_WITH_NAMESPACE_PREFIXES, prop, k=3)

# Read in sample inputs
urls = [
    "https://raw.githubusercontent.com/lmtoo/alfresco-war/38cded3f4bf91640b6c146aca2e71f9911fc93fe/src/main/resources/alfresco/templates/new-user-templates.xml",
    "https://raw.githubusercontent.com/jacklovett/Tic-Stat-Toe/298550149d74efefad170bb986943adde856f87c/pom.xml",
    "https://raw.githubusercontent.com/SaiZawMyint/java_training_assignment/c055525604ff98df06d03023bf654cad2ea528a5/Assignment1/src/main/webapp/WEB-INF/web.xml",
    "https://raw.githubusercontent.com/bestand/j-gym/90b987233856e4f9751b11ddd0b639153b1b884e/sitemap.xml",
    "https://raw.githubusercontent.com/bagasbest/BeeApp/bcb238637d4c845e24ba87a0fe8b7a0d5a101c0c/app/src/main/res/layout/item_income.xml",
]

positive_trees = []
reduced_trees = []

for url in urls:
    file_name = url.split("/")[-1]
    tree_file = f"{dirname}/inputs/{file_name}.tree"
    reduced_tree_file = f"{dirname}/inputs/{file_name}.reduced.tree"

    if os.path.isfile(tree_file) and os.path.isfile(reduced_tree_file):
        with open(tree_file, 'rb') as file:
            positive_trees.append(pickle.loads(file.read()))

        with open(reduced_tree_file, 'rb') as file:
            reduced_trees.append(pickle.loads(file.read()))

        continue

    with urllib.request.urlopen(url) as f:
        # The XML grammar is a little simplified, so we remove some elements.
        xml_doc: str = f.read().decode('utf-8').strip()
        # Compress whitespace
        xml_doc = re.sub(r"\s+", " ", xml_doc)
        xml_doc = xml_doc.replace("> <", "><")
        xml_doc = xml_doc.replace(" />", "/>")
        xml_doc = re.sub(r"<!DOCTYPE[^>]+?>", "", xml_doc)  # Doctype declaration
        xml_doc = re.sub(r"<\?xml[^>]+?\?>", "", xml_doc)  # XML version declaration
        xml_doc = re.sub(r"<!--.*?-->", "", xml_doc)  # Comments

        # Make sure we still have valid XML.
        try:
            ET.fromstring(xml_doc)
        except Exception as err:
            assert False, str(err)

    sample_tree = language.DerivationTree.from_parse_tree(list(parser.parse(xml_doc))[0])
    reduced_tree = reducer.reduce_by_smallest_subtree_replacement(sample_tree)

    with open(tree_file, 'wb') as file:
        file.write(pickle.dumps(sample_tree))

    with open(reduced_tree_file, 'wb') as file:
        file.write(pickle.dumps(reduced_tree))

    positive_trees.append(sample_tree)
    reduced_trees.append(reduced_tree)

assert len(positive_trees) == len(urls)
assert len(reduced_trees) == len(urls)

# Learn invariants
result = InvariantLearner(
    XML_GRAMMAR_WITH_NAMESPACE_PREFIXES,
    prop,
    activated_patterns={
        # "Balance",
        "Def-Use (XML-Tag Strict)",
    },
    positive_examples=reduced_trees[:3],
    generate_new_learning_samples=False,
    do_generate_more_inputs=True,
    mexpr_expansion_limit=3,
    reduce_inputs_for_learning=False,
    reduce_all_inputs=True,
    target_number_negative_samples=10,
    target_number_positive_samples=3,
    filter_inputs_for_learning_by_kpaths=False,
    min_precision=.3,  # Low precision needed because inv holds trivially for self-closing tags
    # We have to reduce the search space. With all nonterminals included, we obtain
    # hundreds of thousands of instantiations for Def-Use (XML-Tag). Even with basic character
    # nonterminals excluded, it's still tens of thousands. We also exclude <text> (likely irrelevant)
    # and opening / closing tag nonterminals, since the identifiers inside are likely what matters.
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
).learn_invariants()

print("\n".join(map(lambda p: f"{p[1]}: " + ISLaUnparser(p[0]).unparse(), result.items())))

best_invariant, (precision, recall) = next(iter(result.items()))
print(f"Best invariant (*estimated* precision {precision:.2f}, recall {recall:.2f}):")
print(ISLaUnparser(best_invariant).unparse())
