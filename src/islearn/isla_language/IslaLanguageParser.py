# Generated from IslaLanguage.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,40,140,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,1,0,3,
        0,14,8,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,
        1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
        5,2,78,8,2,10,2,12,2,81,9,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,90,8,
        2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,5,
        2,107,8,2,10,2,12,2,110,9,2,1,3,1,3,1,3,1,3,3,3,116,8,3,1,4,1,4,
        1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,4,4,130,8,4,11,4,12,4,131,
        1,4,1,4,3,4,136,8,4,1,5,1,5,1,5,0,1,4,6,0,2,4,6,8,10,0,2,3,0,7,7,
        23,26,31,38,3,0,18,18,21,22,27,29,160,0,13,1,0,0,0,2,17,1,0,0,0,
        4,89,1,0,0,0,6,115,1,0,0,0,8,135,1,0,0,0,10,137,1,0,0,0,12,14,3,
        2,1,0,13,12,1,0,0,0,13,14,1,0,0,0,14,15,1,0,0,0,15,16,3,4,2,0,16,
        1,1,0,0,0,17,18,5,1,0,0,18,19,5,28,0,0,19,20,5,2,0,0,20,21,3,6,3,
        0,21,22,5,3,0,0,22,3,1,0,0,0,23,24,6,2,-1,0,24,25,5,4,0,0,25,26,
        3,6,3,0,26,27,5,28,0,0,27,28,5,5,0,0,28,29,5,28,0,0,29,30,5,2,0,
        0,30,31,3,4,2,15,31,90,1,0,0,0,32,33,5,6,0,0,33,34,3,6,3,0,34,35,
        5,28,0,0,35,36,5,5,0,0,36,37,5,28,0,0,37,38,5,2,0,0,38,39,3,4,2,
        14,39,90,1,0,0,0,40,41,5,4,0,0,41,42,3,6,3,0,42,43,5,28,0,0,43,44,
        5,7,0,0,44,45,5,27,0,0,45,46,5,5,0,0,46,47,5,28,0,0,47,48,5,2,0,
        0,48,49,3,4,2,13,49,90,1,0,0,0,50,51,5,6,0,0,51,52,3,6,3,0,52,53,
        5,28,0,0,53,54,5,7,0,0,54,55,5,27,0,0,55,56,5,5,0,0,56,57,5,28,0,
        0,57,58,5,2,0,0,58,59,3,4,2,12,59,90,1,0,0,0,60,61,5,6,0,0,61,62,
        5,8,0,0,62,63,5,28,0,0,63,64,5,2,0,0,64,90,3,4,2,11,65,66,5,4,0,
        0,66,67,5,8,0,0,67,68,5,28,0,0,68,69,5,2,0,0,69,90,3,4,2,10,70,71,
        5,9,0,0,71,90,3,4,2,9,72,73,5,28,0,0,73,74,5,15,0,0,74,79,3,10,5,
        0,75,76,5,16,0,0,76,78,3,10,5,0,77,75,1,0,0,0,78,81,1,0,0,0,79,77,
        1,0,0,0,79,80,1,0,0,0,80,82,1,0,0,0,81,79,1,0,0,0,82,83,5,17,0,0,
        83,90,1,0,0,0,84,90,3,8,4,0,85,86,5,15,0,0,86,87,3,4,2,0,87,88,5,
        17,0,0,88,90,1,0,0,0,89,23,1,0,0,0,89,32,1,0,0,0,89,40,1,0,0,0,89,
        50,1,0,0,0,89,60,1,0,0,0,89,65,1,0,0,0,89,70,1,0,0,0,89,72,1,0,0,
        0,89,84,1,0,0,0,89,85,1,0,0,0,90,108,1,0,0,0,91,92,10,8,0,0,92,93,
        5,10,0,0,93,107,3,4,2,9,94,95,10,7,0,0,95,96,5,11,0,0,96,107,3,4,
        2,8,97,98,10,6,0,0,98,99,5,12,0,0,99,107,3,4,2,7,100,101,10,5,0,
        0,101,102,5,13,0,0,102,107,3,4,2,6,103,104,10,4,0,0,104,105,5,14,
        0,0,105,107,3,4,2,5,106,91,1,0,0,0,106,94,1,0,0,0,106,97,1,0,0,0,
        106,100,1,0,0,0,106,103,1,0,0,0,107,110,1,0,0,0,108,106,1,0,0,0,
        108,109,1,0,0,0,109,5,1,0,0,0,110,108,1,0,0,0,111,112,5,38,0,0,112,
        113,5,28,0,0,113,116,5,37,0,0,114,116,5,18,0,0,115,111,1,0,0,0,115,
        114,1,0,0,0,116,7,1,0,0,0,117,136,5,19,0,0,118,136,5,20,0,0,119,
        136,5,29,0,0,120,136,5,28,0,0,121,136,5,27,0,0,122,136,5,18,0,0,
        123,136,5,21,0,0,124,136,5,22,0,0,125,136,7,0,0,0,126,127,5,15,0,
        0,127,129,3,8,4,0,128,130,3,8,4,0,129,128,1,0,0,0,130,131,1,0,0,
        0,131,129,1,0,0,0,131,132,1,0,0,0,132,133,1,0,0,0,133,134,5,17,0,
        0,134,136,1,0,0,0,135,117,1,0,0,0,135,118,1,0,0,0,135,119,1,0,0,
        0,135,120,1,0,0,0,135,121,1,0,0,0,135,122,1,0,0,0,135,123,1,0,0,
        0,135,124,1,0,0,0,135,125,1,0,0,0,135,126,1,0,0,0,136,9,1,0,0,0,
        137,138,7,1,0,0,138,11,1,0,0,0,8,13,79,89,106,108,115,131,135
    ]

class IslaLanguageParser ( Parser ):

    grammarFileName = "IslaLanguage.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'const'", "':'", "';'", "'forall'", "'in'", 
                     "'exists'", "'='", "'int'", "'not'", "'and'", "'or'", 
                     "'xor'", "'implies'", "'iff'", "'('", "','", "')'", 
                     "'<?NONTERMINAL>'", "'true'", "'false'", "'<?STRING>'", 
                     "'<?DSTRINGS>'", "'re.*'", "'re.++'", "'re.+'", "'str.++'", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "'/'", "'*'", "'+'", "'-'", "'>='", "'<='", "'>'", 
                     "'<'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "STRING", "ID", 
                      "INT", "ESC", "DIV", "MUL", "PLUS", "MINUS", "GEQ", 
                      "LEQ", "GT", "LT", "WS", "LINE_COMMENT" ]

    RULE_start = 0
    RULE_constDecl = 1
    RULE_formula = 2
    RULE_varType = 3
    RULE_sexpr = 4
    RULE_predicateArg = 5

    ruleNames =  [ "start", "constDecl", "formula", "varType", "sexpr", 
                   "predicateArg" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    STRING=27
    ID=28
    INT=29
    ESC=30
    DIV=31
    MUL=32
    PLUS=33
    MINUS=34
    GEQ=35
    LEQ=36
    GT=37
    LT=38
    WS=39
    LINE_COMMENT=40

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class StartContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)


        def constDecl(self):
            return self.getTypedRuleContext(IslaLanguageParser.ConstDeclContext,0)


        def getRuleIndex(self):
            return IslaLanguageParser.RULE_start

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStart" ):
                listener.enterStart(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStart" ):
                listener.exitStart(self)




    def start(self):

        localctx = IslaLanguageParser.StartContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_start)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 13
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==IslaLanguageParser.T__0:
                self.state = 12
                self.constDecl()


            self.state = 15
            self.formula(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ConstDeclContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)

        def varType(self):
            return self.getTypedRuleContext(IslaLanguageParser.VarTypeContext,0)


        def getRuleIndex(self):
            return IslaLanguageParser.RULE_constDecl

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConstDecl" ):
                listener.enterConstDecl(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConstDecl" ):
                listener.exitConstDecl(self)




    def constDecl(self):

        localctx = IslaLanguageParser.ConstDeclContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_constDecl)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 17
            self.match(IslaLanguageParser.T__0)
            self.state = 18
            self.match(IslaLanguageParser.ID)
            self.state = 19
            self.match(IslaLanguageParser.T__1)
            self.state = 20
            self.varType()
            self.state = 21
            self.match(IslaLanguageParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FormulaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return IslaLanguageParser.RULE_formula

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class ExistsMexprContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.varId = None # Token
            self.inId = None # Token
            self.copyFrom(ctx)

        def varType(self):
            return self.getTypedRuleContext(IslaLanguageParser.VarTypeContext,0)

        def STRING(self):
            return self.getToken(IslaLanguageParser.STRING, 0)
        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(IslaLanguageParser.ID)
            else:
                return self.getToken(IslaLanguageParser.ID, i)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExistsMexpr" ):
                listener.enterExistsMexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExistsMexpr" ):
                listener.exitExistsMexpr(self)


    class NegationContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterNegation" ):
                listener.enterNegation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitNegation" ):
                listener.exitNegation(self)


    class ImplicationContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.FormulaContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterImplication" ):
                listener.enterImplication(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitImplication" ):
                listener.exitImplication(self)


    class ForallMexprContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.varId = None # Token
            self.inId = None # Token
            self.copyFrom(ctx)

        def varType(self):
            return self.getTypedRuleContext(IslaLanguageParser.VarTypeContext,0)

        def STRING(self):
            return self.getToken(IslaLanguageParser.STRING, 0)
        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(IslaLanguageParser.ID)
            else:
                return self.getToken(IslaLanguageParser.ID, i)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterForallMexpr" ):
                listener.enterForallMexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitForallMexpr" ):
                listener.exitForallMexpr(self)


    class ExistsIntContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)
        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExistsInt" ):
                listener.enterExistsInt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExistsInt" ):
                listener.exitExistsInt(self)


    class DisjunctionContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.FormulaContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDisjunction" ):
                listener.enterDisjunction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDisjunction" ):
                listener.exitDisjunction(self)


    class PredicateAtomContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)
        def predicateArg(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.PredicateArgContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.PredicateArgContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredicateAtom" ):
                listener.enterPredicateAtom(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredicateAtom" ):
                listener.exitPredicateAtom(self)


    class SMTFormulaContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def sexpr(self):
            return self.getTypedRuleContext(IslaLanguageParser.SexprContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSMTFormula" ):
                listener.enterSMTFormula(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSMTFormula" ):
                listener.exitSMTFormula(self)


    class EquivalenceContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.FormulaContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEquivalence" ):
                listener.enterEquivalence(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEquivalence" ):
                listener.exitEquivalence(self)


    class ExistsContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.varId = None # Token
            self.inId = None # Token
            self.copyFrom(ctx)

        def varType(self):
            return self.getTypedRuleContext(IslaLanguageParser.VarTypeContext,0)

        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(IslaLanguageParser.ID)
            else:
                return self.getToken(IslaLanguageParser.ID, i)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExists" ):
                listener.enterExists(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExists" ):
                listener.exitExists(self)


    class ConjunctionContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.FormulaContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterConjunction" ):
                listener.enterConjunction(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitConjunction" ):
                listener.exitConjunction(self)


    class ParFormulaContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParFormula" ):
                listener.enterParFormula(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParFormula" ):
                listener.exitParFormula(self)


    class ForallContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.varId = None # Token
            self.inId = None # Token
            self.copyFrom(ctx)

        def varType(self):
            return self.getTypedRuleContext(IslaLanguageParser.VarTypeContext,0)

        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(IslaLanguageParser.ID)
            else:
                return self.getToken(IslaLanguageParser.ID, i)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterForall" ):
                listener.enterForall(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitForall" ):
                listener.exitForall(self)


    class ExclusiveOrContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def formula(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.FormulaContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExclusiveOr" ):
                listener.enterExclusiveOr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExclusiveOr" ):
                listener.exitExclusiveOr(self)


    class ForallIntContext(FormulaContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.FormulaContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)
        def formula(self):
            return self.getTypedRuleContext(IslaLanguageParser.FormulaContext,0)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterForallInt" ):
                listener.enterForallInt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitForallInt" ):
                listener.exitForallInt(self)



    def formula(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = IslaLanguageParser.FormulaContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_formula, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 89
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                localctx = IslaLanguageParser.ForallContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 24
                self.match(IslaLanguageParser.T__3)
                self.state = 25
                self.varType()
                self.state = 26
                localctx.varId = self.match(IslaLanguageParser.ID)
                self.state = 27
                self.match(IslaLanguageParser.T__4)
                self.state = 28
                localctx.inId = self.match(IslaLanguageParser.ID)
                self.state = 29
                self.match(IslaLanguageParser.T__1)
                self.state = 30
                self.formula(15)
                pass

            elif la_ == 2:
                localctx = IslaLanguageParser.ExistsContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 32
                self.match(IslaLanguageParser.T__5)
                self.state = 33
                self.varType()
                self.state = 34
                localctx.varId = self.match(IslaLanguageParser.ID)
                self.state = 35
                self.match(IslaLanguageParser.T__4)
                self.state = 36
                localctx.inId = self.match(IslaLanguageParser.ID)
                self.state = 37
                self.match(IslaLanguageParser.T__1)
                self.state = 38
                self.formula(14)
                pass

            elif la_ == 3:
                localctx = IslaLanguageParser.ForallMexprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 40
                self.match(IslaLanguageParser.T__3)
                self.state = 41
                self.varType()
                self.state = 42
                localctx.varId = self.match(IslaLanguageParser.ID)
                self.state = 43
                self.match(IslaLanguageParser.T__6)
                self.state = 44
                self.match(IslaLanguageParser.STRING)
                self.state = 45
                self.match(IslaLanguageParser.T__4)
                self.state = 46
                localctx.inId = self.match(IslaLanguageParser.ID)
                self.state = 47
                self.match(IslaLanguageParser.T__1)
                self.state = 48
                self.formula(13)
                pass

            elif la_ == 4:
                localctx = IslaLanguageParser.ExistsMexprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 50
                self.match(IslaLanguageParser.T__5)
                self.state = 51
                self.varType()
                self.state = 52
                localctx.varId = self.match(IslaLanguageParser.ID)
                self.state = 53
                self.match(IslaLanguageParser.T__6)
                self.state = 54
                self.match(IslaLanguageParser.STRING)
                self.state = 55
                self.match(IslaLanguageParser.T__4)
                self.state = 56
                localctx.inId = self.match(IslaLanguageParser.ID)
                self.state = 57
                self.match(IslaLanguageParser.T__1)
                self.state = 58
                self.formula(12)
                pass

            elif la_ == 5:
                localctx = IslaLanguageParser.ExistsIntContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 60
                self.match(IslaLanguageParser.T__5)
                self.state = 61
                self.match(IslaLanguageParser.T__7)
                self.state = 62
                self.match(IslaLanguageParser.ID)
                self.state = 63
                self.match(IslaLanguageParser.T__1)
                self.state = 64
                self.formula(11)
                pass

            elif la_ == 6:
                localctx = IslaLanguageParser.ForallIntContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 65
                self.match(IslaLanguageParser.T__3)
                self.state = 66
                self.match(IslaLanguageParser.T__7)
                self.state = 67
                self.match(IslaLanguageParser.ID)
                self.state = 68
                self.match(IslaLanguageParser.T__1)
                self.state = 69
                self.formula(10)
                pass

            elif la_ == 7:
                localctx = IslaLanguageParser.NegationContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 70
                self.match(IslaLanguageParser.T__8)
                self.state = 71
                self.formula(9)
                pass

            elif la_ == 8:
                localctx = IslaLanguageParser.PredicateAtomContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 72
                self.match(IslaLanguageParser.ID)
                self.state = 73
                self.match(IslaLanguageParser.T__14)
                self.state = 74
                self.predicateArg()
                self.state = 79
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==IslaLanguageParser.T__15:
                    self.state = 75
                    self.match(IslaLanguageParser.T__15)
                    self.state = 76
                    self.predicateArg()
                    self.state = 81
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 82
                self.match(IslaLanguageParser.T__16)
                pass

            elif la_ == 9:
                localctx = IslaLanguageParser.SMTFormulaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 84
                self.sexpr()
                pass

            elif la_ == 10:
                localctx = IslaLanguageParser.ParFormulaContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 85
                self.match(IslaLanguageParser.T__14)
                self.state = 86
                self.formula(0)
                self.state = 87
                self.match(IslaLanguageParser.T__16)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 108
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,4,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 106
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
                    if la_ == 1:
                        localctx = IslaLanguageParser.ConjunctionContext(self, IslaLanguageParser.FormulaContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 91
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 92
                        self.match(IslaLanguageParser.T__9)
                        self.state = 93
                        self.formula(9)
                        pass

                    elif la_ == 2:
                        localctx = IslaLanguageParser.DisjunctionContext(self, IslaLanguageParser.FormulaContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 94
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 95
                        self.match(IslaLanguageParser.T__10)
                        self.state = 96
                        self.formula(8)
                        pass

                    elif la_ == 3:
                        localctx = IslaLanguageParser.ExclusiveOrContext(self, IslaLanguageParser.FormulaContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 97
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 98
                        self.match(IslaLanguageParser.T__11)
                        self.state = 99
                        self.formula(7)
                        pass

                    elif la_ == 4:
                        localctx = IslaLanguageParser.ImplicationContext(self, IslaLanguageParser.FormulaContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 100
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 101
                        self.match(IslaLanguageParser.T__12)
                        self.state = 102
                        self.formula(6)
                        pass

                    elif la_ == 5:
                        localctx = IslaLanguageParser.EquivalenceContext(self, IslaLanguageParser.FormulaContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_formula)
                        self.state = 103
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 104
                        self.match(IslaLanguageParser.T__13)
                        self.state = 105
                        self.formula(5)
                        pass

             
                self.state = 110
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,4,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class VarTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LT(self):
            return self.getToken(IslaLanguageParser.LT, 0)

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)

        def GT(self):
            return self.getToken(IslaLanguageParser.GT, 0)

        def getRuleIndex(self):
            return IslaLanguageParser.RULE_varType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVarType" ):
                listener.enterVarType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVarType" ):
                listener.exitVarType(self)




    def varType(self):

        localctx = IslaLanguageParser.VarTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_varType)
        try:
            self.state = 115
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [IslaLanguageParser.LT]:
                self.enterOuterAlt(localctx, 1)
                self.state = 111
                self.match(IslaLanguageParser.LT)
                self.state = 112
                self.match(IslaLanguageParser.ID)
                self.state = 113
                self.match(IslaLanguageParser.GT)
                pass
            elif token in [IslaLanguageParser.T__17]:
                self.enterOuterAlt(localctx, 2)
                self.state = 114
                self.match(IslaLanguageParser.T__17)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return IslaLanguageParser.RULE_sexpr

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class SexprNonterminalStringPhContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprNonterminalStringPh" ):
                listener.enterSexprNonterminalStringPh(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprNonterminalStringPh" ):
                listener.exitSexprNonterminalStringPh(self)


    class SexprStrContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def STRING(self):
            return self.getToken(IslaLanguageParser.STRING, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprStr" ):
                listener.enterSexprStr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprStr" ):
                listener.exitSexprStr(self)


    class SexprNumContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def INT(self):
            return self.getToken(IslaLanguageParser.INT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprNum" ):
                listener.enterSexprNum(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprNum" ):
                listener.exitSexprNum(self)


    class SexprOpContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def DIV(self):
            return self.getToken(IslaLanguageParser.DIV, 0)
        def MUL(self):
            return self.getToken(IslaLanguageParser.MUL, 0)
        def PLUS(self):
            return self.getToken(IslaLanguageParser.PLUS, 0)
        def MINUS(self):
            return self.getToken(IslaLanguageParser.MINUS, 0)
        def GEQ(self):
            return self.getToken(IslaLanguageParser.GEQ, 0)
        def LEQ(self):
            return self.getToken(IslaLanguageParser.LEQ, 0)
        def GT(self):
            return self.getToken(IslaLanguageParser.GT, 0)
        def LT(self):
            return self.getToken(IslaLanguageParser.LT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprOp" ):
                listener.enterSexprOp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprOp" ):
                listener.exitSexprOp(self)


    class SexprTrueContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprTrue" ):
                listener.enterSexprTrue(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprTrue" ):
                listener.exitSexprTrue(self)


    class SexprStringPhContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprStringPh" ):
                listener.enterSexprStringPh(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprStringPh" ):
                listener.exitSexprStringPh(self)


    class SexprDisjStringsPhContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprDisjStringsPh" ):
                listener.enterSexprDisjStringsPh(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprDisjStringsPh" ):
                listener.exitSexprDisjStringsPh(self)


    class SexprFalseContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprFalse" ):
                listener.enterSexprFalse(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprFalse" ):
                listener.exitSexprFalse(self)


    class SepxrAppContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.op = None # SexprContext
            self.copyFrom(ctx)

        def sexpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(IslaLanguageParser.SexprContext)
            else:
                return self.getTypedRuleContext(IslaLanguageParser.SexprContext,i)


        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSepxrApp" ):
                listener.enterSepxrApp(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSepxrApp" ):
                listener.exitSepxrApp(self)


    class SexprIdContext(SexprContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a IslaLanguageParser.SexprContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSexprId" ):
                listener.enterSexprId(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSexprId" ):
                listener.exitSexprId(self)



    def sexpr(self):

        localctx = IslaLanguageParser.SexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_sexpr)
        self._la = 0 # Token type
        try:
            self.state = 135
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [IslaLanguageParser.T__18]:
                localctx = IslaLanguageParser.SexprTrueContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 117
                self.match(IslaLanguageParser.T__18)
                pass
            elif token in [IslaLanguageParser.T__19]:
                localctx = IslaLanguageParser.SexprFalseContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 118
                self.match(IslaLanguageParser.T__19)
                pass
            elif token in [IslaLanguageParser.INT]:
                localctx = IslaLanguageParser.SexprNumContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 119
                self.match(IslaLanguageParser.INT)
                pass
            elif token in [IslaLanguageParser.ID]:
                localctx = IslaLanguageParser.SexprIdContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 120
                self.match(IslaLanguageParser.ID)
                pass
            elif token in [IslaLanguageParser.STRING]:
                localctx = IslaLanguageParser.SexprStrContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 121
                self.match(IslaLanguageParser.STRING)
                pass
            elif token in [IslaLanguageParser.T__17]:
                localctx = IslaLanguageParser.SexprNonterminalStringPhContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 122
                self.match(IslaLanguageParser.T__17)
                pass
            elif token in [IslaLanguageParser.T__20]:
                localctx = IslaLanguageParser.SexprStringPhContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 123
                self.match(IslaLanguageParser.T__20)
                pass
            elif token in [IslaLanguageParser.T__21]:
                localctx = IslaLanguageParser.SexprDisjStringsPhContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 124
                self.match(IslaLanguageParser.T__21)
                pass
            elif token in [IslaLanguageParser.T__6, IslaLanguageParser.T__22, IslaLanguageParser.T__23, IslaLanguageParser.T__24, IslaLanguageParser.T__25, IslaLanguageParser.DIV, IslaLanguageParser.MUL, IslaLanguageParser.PLUS, IslaLanguageParser.MINUS, IslaLanguageParser.GEQ, IslaLanguageParser.LEQ, IslaLanguageParser.GT, IslaLanguageParser.LT]:
                localctx = IslaLanguageParser.SexprOpContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 125
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << IslaLanguageParser.T__6) | (1 << IslaLanguageParser.T__22) | (1 << IslaLanguageParser.T__23) | (1 << IslaLanguageParser.T__24) | (1 << IslaLanguageParser.T__25) | (1 << IslaLanguageParser.DIV) | (1 << IslaLanguageParser.MUL) | (1 << IslaLanguageParser.PLUS) | (1 << IslaLanguageParser.MINUS) | (1 << IslaLanguageParser.GEQ) | (1 << IslaLanguageParser.LEQ) | (1 << IslaLanguageParser.GT) | (1 << IslaLanguageParser.LT))) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                pass
            elif token in [IslaLanguageParser.T__14]:
                localctx = IslaLanguageParser.SepxrAppContext(self, localctx)
                self.enterOuterAlt(localctx, 10)
                self.state = 126
                self.match(IslaLanguageParser.T__14)
                self.state = 127
                localctx.op = self.sexpr()
                self.state = 129 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while True:
                    self.state = 128
                    self.sexpr()
                    self.state = 131 
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << IslaLanguageParser.T__6) | (1 << IslaLanguageParser.T__14) | (1 << IslaLanguageParser.T__17) | (1 << IslaLanguageParser.T__18) | (1 << IslaLanguageParser.T__19) | (1 << IslaLanguageParser.T__20) | (1 << IslaLanguageParser.T__21) | (1 << IslaLanguageParser.T__22) | (1 << IslaLanguageParser.T__23) | (1 << IslaLanguageParser.T__24) | (1 << IslaLanguageParser.T__25) | (1 << IslaLanguageParser.STRING) | (1 << IslaLanguageParser.ID) | (1 << IslaLanguageParser.INT) | (1 << IslaLanguageParser.DIV) | (1 << IslaLanguageParser.MUL) | (1 << IslaLanguageParser.PLUS) | (1 << IslaLanguageParser.MINUS) | (1 << IslaLanguageParser.GEQ) | (1 << IslaLanguageParser.LEQ) | (1 << IslaLanguageParser.GT) | (1 << IslaLanguageParser.LT))) != 0)):
                        break

                self.state = 133
                self.match(IslaLanguageParser.T__16)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PredicateArgContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(IslaLanguageParser.ID, 0)

        def INT(self):
            return self.getToken(IslaLanguageParser.INT, 0)

        def STRING(self):
            return self.getToken(IslaLanguageParser.STRING, 0)

        def getRuleIndex(self):
            return IslaLanguageParser.RULE_predicateArg

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPredicateArg" ):
                listener.enterPredicateArg(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPredicateArg" ):
                listener.exitPredicateArg(self)




    def predicateArg(self):

        localctx = IslaLanguageParser.PredicateArgContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_predicateArg)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 137
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << IslaLanguageParser.T__17) | (1 << IslaLanguageParser.T__20) | (1 << IslaLanguageParser.T__21) | (1 << IslaLanguageParser.STRING) | (1 << IslaLanguageParser.ID) | (1 << IslaLanguageParser.INT))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[2] = self.formula_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def formula_sempred(self, localctx:FormulaContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 8)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 7)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 4)
         




