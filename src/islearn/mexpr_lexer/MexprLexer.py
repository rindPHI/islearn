# Generated from MexprLexer.g4 by ANTLR 4.10.1
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,14,110,6,-1,6,-1,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,
        4,2,5,7,5,2,6,7,6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,
        12,7,12,2,13,7,13,2,14,7,14,2,15,7,15,1,0,1,0,1,0,1,0,1,1,1,1,1,
        1,1,1,1,2,4,2,45,8,2,11,2,12,2,46,1,3,4,3,50,8,3,11,3,12,3,51,1,
        3,1,3,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,5,1,
        5,1,5,1,5,1,6,1,6,1,6,1,7,1,7,1,8,1,8,1,8,5,8,81,8,8,10,8,12,8,84,
        9,8,1,9,3,9,87,8,9,1,10,1,10,1,11,1,11,1,12,1,12,1,13,4,13,96,8,
        13,11,13,12,13,97,1,13,1,13,1,14,1,14,1,14,1,14,1,15,4,15,107,8,
        15,11,15,12,15,108,0,0,16,3,1,5,2,7,3,9,4,11,5,13,6,15,7,17,8,19,
        9,21,0,23,0,25,10,27,11,29,12,31,13,33,14,3,0,1,2,4,2,0,91,91,123,
        123,4,0,45,46,65,90,95,95,97,122,3,0,9,10,13,13,32,32,1,0,93,93,
        111,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,0,0,0,1,11,1,0,0,0,
        1,13,1,0,0,0,1,15,1,0,0,0,1,17,1,0,0,0,1,19,1,0,0,0,1,25,1,0,0,0,
        1,27,1,0,0,0,1,29,1,0,0,0,2,31,1,0,0,0,2,33,1,0,0,0,3,35,1,0,0,0,
        5,39,1,0,0,0,7,44,1,0,0,0,9,49,1,0,0,0,11,55,1,0,0,0,13,59,1,0,0,
        0,15,72,1,0,0,0,17,75,1,0,0,0,19,77,1,0,0,0,21,86,1,0,0,0,23,88,
        1,0,0,0,25,90,1,0,0,0,27,92,1,0,0,0,29,95,1,0,0,0,31,101,1,0,0,0,
        33,106,1,0,0,0,35,36,5,123,0,0,36,37,1,0,0,0,37,38,6,0,0,0,38,4,
        1,0,0,0,39,40,5,91,0,0,40,41,1,0,0,0,41,42,6,1,1,0,42,6,1,0,0,0,
        43,45,8,0,0,0,44,43,1,0,0,0,45,46,1,0,0,0,46,44,1,0,0,0,46,47,1,
        0,0,0,47,8,1,0,0,0,48,50,5,10,0,0,49,48,1,0,0,0,50,51,1,0,0,0,51,
        49,1,0,0,0,51,52,1,0,0,0,52,53,1,0,0,0,53,54,6,3,2,0,54,10,1,0,0,
        0,55,56,5,125,0,0,56,57,1,0,0,0,57,58,6,4,3,0,58,12,1,0,0,0,59,60,
        5,60,0,0,60,61,5,63,0,0,61,62,5,77,0,0,62,63,5,65,0,0,63,64,5,84,
        0,0,64,65,5,67,0,0,65,66,5,72,0,0,66,67,5,69,0,0,67,68,5,88,0,0,
        68,69,5,80,0,0,69,70,5,82,0,0,70,71,5,40,0,0,71,14,1,0,0,0,72,73,
        5,41,0,0,73,74,5,62,0,0,74,16,1,0,0,0,75,76,5,44,0,0,76,18,1,0,0,
        0,77,82,3,21,9,0,78,81,3,21,9,0,79,81,3,23,10,0,80,78,1,0,0,0,80,
        79,1,0,0,0,81,84,1,0,0,0,82,80,1,0,0,0,82,83,1,0,0,0,83,20,1,0,0,
        0,84,82,1,0,0,0,85,87,7,1,0,0,86,85,1,0,0,0,87,22,1,0,0,0,88,89,
        2,48,57,0,89,24,1,0,0,0,90,91,5,62,0,0,91,26,1,0,0,0,92,93,5,60,
        0,0,93,28,1,0,0,0,94,96,7,2,0,0,95,94,1,0,0,0,96,97,1,0,0,0,97,95,
        1,0,0,0,97,98,1,0,0,0,98,99,1,0,0,0,99,100,6,13,2,0,100,30,1,0,0,
        0,101,102,5,93,0,0,102,103,1,0,0,0,103,104,6,14,3,0,104,32,1,0,0,
        0,105,107,8,3,0,0,106,105,1,0,0,0,107,108,1,0,0,0,108,106,1,0,0,
        0,108,109,1,0,0,0,109,34,1,0,0,0,10,0,1,2,46,51,80,82,86,97,108,
        4,5,1,0,5,2,0,6,0,0,4,0,0
    ]

class MexprLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    VAR_DECL = 1
    OPTIONAL = 2

    BRAOP = 1
    OPTOP = 2
    TEXT = 3
    NL = 4
    BRACL = 5
    PLACEHOLDER_OP = 6
    PLACEHOLDER_CL = 7
    COMMA = 8
    ID = 9
    GT = 10
    LT = 11
    WS = 12
    OPTCL = 13
    OPTTXT = 14

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE", "VAR_DECL", "OPTIONAL" ]

    literalNames = [ "<INVALID>",
            "'{'", "'['", "'}'", "'<?MATCHEXPR('", "')>'", "','", "'>'", 
            "'<'", "']'" ]

    symbolicNames = [ "<INVALID>",
            "BRAOP", "OPTOP", "TEXT", "NL", "BRACL", "PLACEHOLDER_OP", "PLACEHOLDER_CL", 
            "COMMA", "ID", "GT", "LT", "WS", "OPTCL", "OPTTXT" ]

    ruleNames = [ "BRAOP", "OPTOP", "TEXT", "NL", "BRACL", "PLACEHOLDER_OP", 
                  "PLACEHOLDER_CL", "COMMA", "ID", "ID_LETTER", "DIGIT", 
                  "GT", "LT", "WS", "OPTCL", "OPTTXT" ]

    grammarFileName = "MexprLexer.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


