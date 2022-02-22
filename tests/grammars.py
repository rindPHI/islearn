import string

from fuzzingbook.Grammars import srange

# NOTE: To make this a PEG grammar, we need to escape single quotes within
#       multiline literal strings. So, a multiline string with """ is the
#       same as a multiline string with ''', just with different quotation marks.
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
    "<value>": ["<string>", "<integer>", "<floating_point>", "<bool>", "<date_time>", "<array>", "<inline_table>"],
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
