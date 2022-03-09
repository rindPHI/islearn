import string

from fuzzingbook.Grammars import srange, CHARACTERS_WITHOUT_QUOTE
# NOTE: To make this a PEG grammar, we need to escape single quotes within
#       multiline literal strings. So, a multiline string with """ is the
#       same as a multiline string with ''', just with different quotation marks.
from isla.type_defs import Grammar

toml_grammar = {
    "<start>": ["<document>"],
    "<document>": ["<expressions>"],
    "<expressions>": ["<expression><NL><expressions>", "<expression>"],
    "<expression>": ["<key_value><comment>", "<table><comment>", "<comment>"],
    "<comment>": ["# <STR_NO_NL>", ""],
    "<STR_NO_NL>": ["<STR_NO_NL_CHARS><STR_NO_NL>", ""],
    "<STR_NO_NL_CHARS>": list(set(srange(string.printable)) - {"\n"}),
    "<key_value>": ["<key> = <value>"],
    "<key>": ["<dotted_key>", "<simple_key>"],
    "<simple_key>": ["<quoted_key>", "<unquoted_key>"],
    "<unquoted_key>": ["<UNQUOTED_KEY>"],
    "<quoted_key>": ["<BASIC_STRING>", "<LITERAL_STRING>"],
    "<dotted_key>": ["<simple_key><dot_simple_key>"],
    "<dot_simple_key>": [".<simple_key><dot_simple_key>", ".<simple_key>"],
    "<value>": ["<string>", "<date_time>", "<LOCAL_DATE>", "<floating_point>", "<integer>", "<bool>", "<array>",
                "<inline_table>"],
    "<string>": ["<ML_BASIC_STRING>", "<BASIC_STRING>", "<ML_LITERAL_STRING>", "<LITERAL_STRING>"],
    "<integer>": ["<DEC_INT>", "<HEX_INT>", "<OCT_INT>", "<BIN_INT>"],
    "<floating_point>": ["<FLOAT>", "<INF>", "<NAN>"],
    "<bool>": ["<BOOLEAN>"],
    "<date_time>": ["<OFFSET_DATE_TIME>", "<LOCAL_DATE_TIME>", "<LOCAL_DATE>", "<LOCAL_TIME>"],
    "<array>": ["[<opt_array_values><comment_or_nl>]"],
    "<opt_array_values>": ["<array_values>", ""],
    "<array_values>": [
        "<comment_or_nl><value><nl_or_comment><opt_comma>",
        "<comment_or_nl><value><nl_or_comment>,<array_values><comment_or_nl>"
    ],
    "<opt_comma>": [",", ""],
    "<comment_or_nl>": ["<COMMENT><NL><comment_or_nl>", "<NL><comment_or_nl>", "<COMMENT><NL>", "<NL>", ""],
    "<nl_or_comment>": ["<NL><COMMENT><nl_or_comment>", "<NL><nl_or_comment>", "<NL><COMMENT>", "<NL>", ""],
    "<table>": ["<standard_table>", "<array_table>"],
    "<standard_table>": ["[<key>]"],
    "<inline_table>": ["{<inline_table_keyvals>}"],
    "<inline_table_keyvals>": ["<inline_table_keyvals_non_empty>", ""],
    "<inline_table_keyvals_non_empty>": [
        "<key> = <value>, <inline_table_keyvals_non_empty>",
        "<key> = <value>"
    ],
    "<array_table>": ["[[<key>]]"],

    "<NL>": ["\n"],
    "<COMMENT>": ["# <STR_NO_NL>"],

    "<DIGIT>": srange(string.digits),
    "<ALPHA>": srange(string.ascii_letters),

    # booleans
    "<BOOLEAN>": ["true", "false"],

    # strings
    "<ESC>": [
        '\\"', "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t", "\\<UNICODE>", "\\<EX_UNICODE>",
    ],
    "<ML_ESC>": ["<ESC>", "\\<opt_carriage_return>\n"],
    "<opt_carriage_return>": ["\r", ""],
    "<UNICODE>": ["u<HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT>"],
    "<EX_UNICODE>": ["U<HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT><HEX_DIGIT>"],
    "<BASIC_STRING>": ['"<esc_or_no_string_endings>"'],
    "<esc_or_no_string_endings>": ["<esc_or_no_string_ending><esc_or_no_string_endings>", ""],
    "<esc_or_no_string_ending>": ["<ESC>", "<no_string_ending>"],
    "<no_string_ending>": list(set(srange(string.printable)) - set(srange('"\\\n'))),
    "<ML_BASIC_STRING>": ['"""<ml_esc_or_no_string_endings>"""'],
    "<ml_esc_or_no_string_endings>": ["<ml_esc_or_no_string_ending><ml_esc_or_no_string_endings>", ""],
    "<ml_esc_or_no_string_ending>": ["<ML_ESC>", "<no_ml_string_ending>"],
    "<no_ml_string_ending>": list(set(srange(string.printable)) - set(srange('"\\'))),
    "<LITERAL_STRING>": ["'<no_literal_string_endings>'"],
    "<no_literal_string_endings>": ["<no_literal_string_ending><no_literal_string_endings>", ""],
    "<no_literal_string_ending>": list(set(srange(string.printable)) - set(srange("'\n"))),
    "<ML_LITERAL_STRING>": ["'''<ml_esc_or_no_literal_string_endings>'''"],
    "<ml_esc_or_no_literal_string_endings>": [
        "<ml_esc_or_no_literal_string_ending><ml_esc_or_no_literal_string_endings>",
        ""],
    "<ml_esc_or_no_literal_string_ending>": ["<ML_ESC>", "<no_ml_literal_string_ending>"],
    "<no_ml_literal_string_ending>": list(set(srange(string.printable)) - set(srange("'\\"))),
    # "<any_chars>": ["<any_char><any_chars>", ""],
    # "<any_char>": srange(string.printable),
    # floating point numbers
    "<EXP>": ["<e><opt_plusminus><ZERO_PREFIXABLE_INT>"],
    "<e>": ["e", "E"],
    "<opt_plusminus>": ["<plusminus>", ""],
    "<plusminus>": ["+", "-"],
    "<ZERO_PREFIXABLE_INT>": ["<DIGIT><DIGIT_OR_UNDERSCORES>"],
    "<DIGIT_OR_UNDERSCORES>": [
        "<DIGIT_OR_UNDERSCORE><DIGIT_OR_UNDERSCORES>",
        "<DIGIT_OR_UNDERSCORE>",
        ""
    ],
    "<DIGIT_OR_UNDERSCORE>": ["<DIGIT>", "_<DIGIT>"],
    "<FRAC>": [".<ZERO_PREFIXABLE_INT>"],
    "<FLOAT>": ["<DEC_INT><FRAC><EXP>", "<DEC_INT><FRAC>", "<DEC_INT><EXP>"],
    "<INF>": ["<opt_plusminus>inf"],
    "<NAN>": ["<opt_plusminus>nan"],
    # integers
    "<HEX_DIGIT>": srange("abcdefABCDEF") + ["<DIGIT>"],
    "<DIGIT_1_9>": srange("123456789"),
    "<DIGIT_0_7>": srange("01234567"),
    "<DIGIT_0_1>": ["0", "1"],
    "<DEC_INT>": [
        "<opt_plusminus><DIGIT_1_9><DIGIT_OR_UNDERSCORE><DIGIT_OR_UNDERSCORES>",
        "<opt_plusminus><DIGIT>"
    ],
    "<HEX_INT>": ["0x<HEX_DIGIT><HEX_DIGIT_OR_UNDERSCORES>"],
    "<HEX_DIGIT_OR_UNDERSCORES>": [
        "<HEX_DIGIT_OR_UNDERSCORE><HEX_DIGIT_OR_UNDERSCORES>",
        "<HEX_DIGIT_OR_UNDERSCORE>",
        ""
    ],
    "<HEX_DIGIT_OR_UNDERSCORE>": ["<HEX_DIGIT>", "_<HEX_DIGIT>"],
    "<OCT_INT>": ["0o<DIGIT_0_7><DIGIT_0_7_OR_UNDERSCORES>"],
    "<DIGIT_0_7_OR_UNDERSCORES>": [
        "<DIGIT_0_7_OR_UNDERSCORE><DIGIT_0_7_OR_UNDERSCORES>",
        "<DIGIT_0_7_OR_UNDERSCORE>",
        ""
    ],
    "<DIGIT_0_7_OR_UNDERSCORE>": ["<DIGIT_0_7>", "_<DIGIT_0_7>"],
    "<BIN_INT>": ["0b<DIGIT_0_1><DIGIT_0_1_OR_UNDERSCORES>"],
    "<DIGIT_0_1_OR_UNDERSCORES>": [
        "<DIGIT_0_1_OR_UNDERSCORE><DIGIT_0_1_OR_UNDERSCORES>",
        "<DIGIT_0_1_OR_UNDERSCORE>",
        ""
    ],
    "<DIGIT_0_1_OR_UNDERSCORE>": ["<DIGIT_0_1>", "_<DIGIT_0_1>"],
    # dates
    "<YEAR>": ["<DIGIT><DIGIT><DIGIT><DIGIT>"],
    "<MONTH>": ["<DIGIT><DIGIT>"],
    "<DAY>": ["<DIGIT><DIGIT>"],
    "<DELIM>": ["T", "t", " "],
    "<HOUR>": ["<DIGIT><DIGIT>"],
    "<MINUTE>": ["<DIGIT><DIGIT>"],
    "<SECOND>": ["<DIGIT><DIGIT>"],
    "<SECFRAC>": [".<DIGITS>"],
    "<DIGITS>": ["<DIGIT>", "<DIGIT><DIGITS>"],
    "<NUMOFFSET>": ["<plusminus><HOUR>:<MINUTE>"],
    "<OFFSET>": ["Z", "<NUMOFFSET>"],
    "<PARTIAL_TIME>": ["<HOUR>:<MINUTE>:<SECOND>", "<HOUR>:<MINUTE>:<SECOND><SECFRAC>"],
    "<FULL_DATE>": ["<YEAR>-<MONTH>-<DAY>"],
    "<FULL_TIME>": ["<PARTIAL_TIME><OFFSET>"],
    "<OFFSET_DATE_TIME>": ["<FULL_DATE><DELIM><FULL_TIME>"],
    "<LOCAL_DATE_TIME>": ["<FULL_DATE><DELIM><PARTIAL_TIME>"],
    "<LOCAL_DATE>": ["<FULL_DATE>"],
    "<LOCAL_TIME>": ["<PARTIAL_TIME>"],
    # keys
    "<UNQUOTED_KEY>": ["<UNQUOTED_KEY_CHARS>"],
    "<UNQUOTED_KEY_CHARS>": ["<UNQUOTED_KEY_CHAR><UNQUOTED_KEY_CHARS>", "<UNQUOTED_KEY_CHAR>"],
    "<UNQUOTED_KEY_CHAR>": ["<ALPHA>", "<DIGIT>", "-", "_"]
}

JSON_GRAMMAR: Grammar = {
    "<start>": ["<json>"],
    "<json>": ["<element>"],
    "<element>": ["<ws><value><ws>"],
    "<value>": ["<object>", "<array>", "<string>", "<number>", "true", "false", "null"],
    "<object>": ["{<members>}", "{<ws>}"],
    "<members>": ["<member>,<members>", "<member>"],
    "<member>": ["<ws><string><ws>:<element>"],
    "<array>": ["[<ws>]", "[<elements>]"],
    "<elements>": ["<element>,<elements>", "<element>"],
    "<string>": ['"' + "<characters>" + '"'],
    "<characters>": ["<character><characters>", ""],
    "<character>": srange(CHARACTERS_WITHOUT_QUOTE),
    "<number>": ["<int><frac><exp>"],
    "<int>": ["-<onenine><digits>", "<onenine><digits>", "-<digit>", "<digit>"],
    "<digits>": ["<digit><digits>", "<digit>"],
    "<digit>": ['0', "<onenine>"],
    "<onenine>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<frac>": [".<digits>", ""],
    "<exp>": ["E<sign><digits>", "e<sign><digits>", ""],
    "<sign>": ['+', '-', ""],
    "<ws>": [" "]
}

ICMP_GRAMMAR = {
    "<start>": ["<icmp_message>"],
    "<icmp_message>": ["<header><payload_data>"],
    "<header>": ["<type><code><checksum><header_data>"],
    "<payload_data>": ["<bytes>"],
    "<type>": ["<byte>"],
    "<code>": ["<byte>"],
    "<checksum>": ["<byte><byte>"],
    "<header_data>": ["<byte><byte><byte><byte>"],
    "<byte>": ["<zerof><zerof> "],
    "<bytes>": ["<byte><bytes>", ""],
    "<zerof>": srange(string.digits + "ABCDEF")
}

# https://en.wikipedia.org/wiki/IPv4#Packet_structure
IPv4_GRAMMAR = {
    "<start>": ["<ip_message>"],
    "<ip_message>": ["<header><data>"],
    "<header>": [  # Each line 4 bytes
        "<version_ihl><dscp_ecn><total_length>"
        "<identification><flags_offset>"
        "<ttl><protocol><header_checksum>"
        "<source_ip>"
        "<dest_ip>"
    ],
    "<data>": ["<bytes>"],
    "<version_ihl>": ["<byte>"],
    "<dscp_ecn>": ["<byte>"],
    "<total_length>": ["<byte><byte>"],
    "<identification>": ["<byte><byte>"],
    "<flags_offset>": ["<byte><byte>"],
    "<ttl>": ["<byte>"],
    "<protocol>": ["<byte>"],
    "<header_checksum>": ["<byte><byte>"],
    "<source_ip>": ["<byte><byte><byte><byte>"],
    "<dest_ip>": ["<byte><byte><byte><byte>"],
    "<byte>": ["<zerof><zerof> "],
    "<bytes>": ["<byte><bytes>", ""],
    "<zerof>": srange(string.digits + "ABCDEF")
}

DOT_GRAMMAR = {
    "<start>": ["<graph>"],
    "<graph>": [
        "<maybe_strict><GRAPH><maybe_space_id><MWSS>{<MWSS><stmt_list><MWSS>}",
        "<maybe_strict><DIGRAPH><maybe_space_id><MWSS>{<MWSS><stmt_list><MWSS>}",
    ],
    "<stmt_list>": ["<stmt><MWSS><maybe_semi><MWSS><stmt_list>", "<stmt><MWSS><maybe_semi>", ""],
    "<stmt>": ["<edge_stmt>", "<attr_stmt>", "<id><MWSS>=<MWSS><id>", "<subgraph>", "<node_stmt>"],
    "<attr_stmt>": ["<GRAPH><MWSS><attr_list>", "<NODE><MWSS><attr_list>", "<EDGE><MWSS><attr_list>"],
    "<maybe_attr_list>": ["<attr_list>", ""],
    "<attr_list>": ["[<MWSS><maybe_a_list><MWSS>]<MWSS><attr_list>", "[<MWSS><maybe_a_list><MWSS>]"],
    "<maybe_a_list>": ["<a_list>", ""],
    "<a_list>": ["<id><maybe_eq_id><MWSS><maybe_comma><MWSS><a_list>", "<id><maybe_eq_id><MWSS><maybe_comma>"],
    "<maybe_eq_id>": ["<MWSS>=<MWSS><id>", ""],
    "<edge_stmt>": [
        "<node_id><MWSS><edgeRHS><MWSS><maybe_attr_list>",
        "<subgraph><MWSS><edgeRHS><MWSS><maybe_attr_list>",
    ],
    "<edgeRHS>": [
        "<edgeop><MWSS><node_id><MWSS><edgeRHS>",
        "<edgeop><MWSS><subgraph><MWSS><edgeRHS>",
        "<edgeop><MWSS><node_id>",
        "<edgeop><MWSS><subgraph>",
    ],
    "<edgeop>": ["->", "--"],
    "<node_stmt>": ["<node_id><MWSS><maybe_attr_list>"],
    "<node_id>": ["<id><MWSS><maybe_port>"],
    "<maybe_port>": ["<port>", ""],
    "<port>": [
        ":<MWSS><id><MWSS>:<MWSS><id>",
        ":<MWSS><id>",
    ],
    "<subgraph>": [
        "<SUBGRAPH><MWSS><maybe_space_id><MWSS>{<MWSS><stmt_list><MWSS>}",
        "{<MWSS><stmt_list><MWSS>}",
    ],
    "<maybe_space_id>": ["<WSS><id>", ""],
    "<id>": ["<STRING>", "<ID>", "<NUMBER>"],
    "<maybe_strict>": ["<STRICT><WSS>", ""],
    "<STRICT>": ["<S><T><R><I><C><T>"],
    "<GRAPH>": ["<G><R><A><P><H>"],
    "<DIGRAPH>": ["<D><I><G><R><A><P><H>"],
    "<NODE>": ["<N><O><D><E>"],
    "<EDGE>": ["<E><D><G><E>"],
    "<SUBGRAPH>": ["<S><U><B><G><R><A><P><H>"],
    "<NUMBER>": [
        "<maybe_minus><DIGITS>.<MAYBE_DIGITS>",
        "<maybe_minus><DIGITS>",
        "<maybe_minus>.<DIGITS>",
    ],
    "<MAYBE_DIGITS>": ["<DIGITS>", ""],
    "<DIGITS>": ["<DIGIT><DIGITS>", "<DIGIT>"],
    "<DIGIT>": srange(string.digits),

    "<STRING>": ['"<esc_or_no_string_endings>"'],
    "<esc_or_no_string_endings>": ["<esc_or_no_string_ending><esc_or_no_string_endings>", ""],
    "<esc_or_no_string_ending>": ['\\"', "<no_string_ending>"],
    "<no_string_ending>": list(set(srange(string.printable)) - set(srange('"\\'))),

    "<ID>": ["<LETTER><LETTER_OR_DIGITS>", ],
    "<LETTER_OR_DIGITS>": [
        "<LETTER><LETTER_OR_DIGITS>",
        "<DIGIT><LETTER_OR_DIGITS>",
        ""
    ],
    "<LETTER>": srange(string.ascii_letters + "_"),
    "<maybe_minus>": ["-", ""],
    "<maybe_comma>": [",", ""],
    "<maybe_semi>": [";", ""],
    "<A>": ["A", "a"],
    "<B>": ["B", "b"],
    "<C>": ["C", "c"],
    "<D>": ["D", "d"],
    "<E>": ["E", "e"],
    "<G>": ["G", "g"],
    "<H>": ["H", "h"],
    "<I>": ["I", "i"],
    "<N>": ["N", "n"],
    "<O>": ["O", "o"],
    "<P>": ["P", "p"],
    "<R>": ["R", "r"],
    "<S>": ["S", "s"],
    "<T>": ["T", "t"],
    "<U>": ["U", "u"],
    "<MWSS>": ["<WSS>", ""],
    "<WSS>": ["<WS><WSS>", "<WS>"],
    "<WS>": [" ", "\t", "\n"]
}

# racket_bsl_grammar = {
# "<program>" : ["<defOrExprs>"],
# "<defOrExpr>": ["<defOrExpr><defOrExprs>", "<defOrExpr>"],
# "<defOrExpr>" : ["<definition>", "<expr>", "<testCase>", "<libraryRequire>"],
#
# "<definition>": [
#     "(define(<name> <NAME>+) <expr>)",
#     "(define <name> <expr>)",
#     "(define <name> (lambda (<NAME>+) <expr>))",
#     "(define-struct <name> (<name>*))",
#     ],
#
#
# expr: [
#     "(<name> <expr>+)",
#     "(<cond> \\([ <expr> <expr> ]\\)+ )",
#     "(<cond> \\([ <expr> <expr> ]\\)* [else <expr>])",
#     "(if <expr> <expr> <expr>)",
#     "(and <expr> <expr>+)",
#     "(<or> <expr> <expr>+)",
#     "’()",
#     "<name>",
#     "NUMBER",
#     "BOOLEAN",
#     "STRING",
#     "CHARACTER",
#     ],
#
# testCase
#     : '(' 'check-expect' expr expr ')'
#     | '(' 'check-random' expr expr ')'
#     | '(' 'check-within' expr expr expr ')'
#     | '(' 'check-member-of' expr expr+ ')'
#     | '(' 'check-satisfied' expr name ')'
#     | '(' 'check-error' expr expr? ')'
#     ;
#
# libraryRequire
#     : '(' 'require' STRING ')'
#     | '(' 'require' name ')'
#     | '(' 'require' '(' name STRING ('(' STRING+ ')')? ')' ')'
#     | '(' 'require' '(' name STRING pkg ')' ')'
#     ;
#
# pkg: '(' STRING STRING NUMBER NUMBER ')'
#    ;
#
# name: SYMBOL
#     | NAME
#     ;
#
# // A symbol is a quote character followed by a name. A symbol is a value, just like 42, '(), or #false.
# SYMBOL
#     : '’' NAME
#     ;
#
# // A name or a variable is a sequence of characters not including a space or one of the following:
# //   " , ' ` ( ) [ ] { } | ; #
# NAME: ([$%&!*+\\^_~]|[--:<-Za-z])+
#     ;
#
# // A number is a number such as 123, 3/2, or 5.5.
# NUMBER
#     : INT
#     | INT '.' [0-9]* [1-9]
#     | INT '/' INT
#     ;
#
# INT: [1-9] [0-9]*
#    | '0'
#    ;
#
# BOOLEAN
#     : '#true'
#     | '#T'
#     | '#t'
#     | '#false'
#     | '#F'
#     | '#f'
#     ;
#
# // A string is a sequence of characters enclosed by a pair of ".
# // Unlike symbols, strings may be split into characters and manipulated by a variety of functions.
# // For example, "abcdef", "This is a string", and "This is a string with \" inside" are all strings.
# STRING
#     : '"' ([ -~])* '"'
#     ;
#
# // A character begins with #\ and has the name of the character.
# // For example, #\a, #\b, and #\space are characters.
# CHARACTER
#     : '#' '\u005C' [A-Za-z0-9]
#     | '#' '\u005C' 'space'
#     ;
#
# LANG: '#lang' ~ ('\n' | '\r')* '\r'? '\n' -> channel (HIDDEN)
#     ;
#
# COMMENT
#    : ';' ~ ('\n' | '\r')* '\r'? '\n' -> channel (HIDDEN)
#    ;
#
# WS: (' ' | '\r' | '\t' | '\u000C' | '\n') -> channel (HIDDEN)
#   ;
# }