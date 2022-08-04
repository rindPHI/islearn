# Generated from MexprParser.g4 by ANTLR 4.10.1
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
        4,1,14,48,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,1,0,4,0,10,8,0,11,0,12,
        0,11,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,1,24,8,1,10,1,12,
        1,27,9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,36,8,1,1,2,1,2,1,2,1,2,
        3,2,42,8,2,1,3,1,3,1,3,1,3,1,3,0,0,4,0,2,4,6,0,0,49,0,9,1,0,0,0,
        2,35,1,0,0,0,4,41,1,0,0,0,6,43,1,0,0,0,8,10,3,2,1,0,9,8,1,0,0,0,
        10,11,1,0,0,0,11,9,1,0,0,0,11,12,1,0,0,0,12,1,1,0,0,0,13,14,5,1,
        0,0,14,15,3,6,3,0,15,16,5,9,0,0,16,17,5,5,0,0,17,36,1,0,0,0,18,19,
        5,1,0,0,19,20,5,6,0,0,20,25,3,4,2,0,21,22,5,8,0,0,22,24,3,4,2,0,
        23,21,1,0,0,0,24,27,1,0,0,0,25,23,1,0,0,0,25,26,1,0,0,0,26,28,1,
        0,0,0,27,25,1,0,0,0,28,29,5,7,0,0,29,30,5,5,0,0,30,36,1,0,0,0,31,
        32,5,2,0,0,32,33,5,14,0,0,33,36,5,13,0,0,34,36,5,3,0,0,35,13,1,0,
        0,0,35,18,1,0,0,0,35,31,1,0,0,0,35,34,1,0,0,0,36,3,1,0,0,0,37,38,
        3,6,3,0,38,39,5,9,0,0,39,42,1,0,0,0,40,42,5,9,0,0,41,37,1,0,0,0,
        41,40,1,0,0,0,42,5,1,0,0,0,43,44,5,11,0,0,44,45,5,9,0,0,45,46,5,
        10,0,0,46,7,1,0,0,0,4,11,25,35,41
    ]

class MexprParser ( Parser ):

    grammarFileName = "MexprParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'{'", "'['", "<INVALID>", "<INVALID>", 
                     "'}'", "'<?MATCHEXPR('", "')>'", "','", "<INVALID>", 
                     "'>'", "'<'", "<INVALID>", "']'" ]

    symbolicNames = [ "<INVALID>", "BRAOP", "OPTOP", "TEXT", "NL", "BRACL", 
                      "PLACEHOLDER_OP", "PLACEHOLDER_CL", "COMMA", "ID", 
                      "GT", "LT", "WS", "OPTCL", "OPTTXT" ]

    RULE_matchExpr = 0
    RULE_matchExprElement = 1
    RULE_mexprPlaceholderParam = 2
    RULE_varType = 3

    ruleNames =  [ "matchExpr", "matchExprElement", "mexprPlaceholderParam", 
                   "varType" ]

    EOF = Token.EOF
    BRAOP=1
    OPTOP=2
    TEXT=3
    NL=4
    BRACL=5
    PLACEHOLDER_OP=6
    PLACEHOLDER_CL=7
    COMMA=8
    ID=9
    GT=10
    LT=11
    WS=12
    OPTCL=13
    OPTTXT=14

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class MatchExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def matchExprElement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(MexprParser.MatchExprElementContext)
            else:
                return self.getTypedRuleContext(MexprParser.MatchExprElementContext,i)


        def getRuleIndex(self):
            return MexprParser.RULE_matchExpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatchExpr" ):
                listener.enterMatchExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatchExpr" ):
                listener.exitMatchExpr(self)




    def matchExpr(self):

        localctx = MexprParser.MatchExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_matchExpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 9 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 8
                self.matchExprElement()
                self.state = 11 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not ((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << MexprParser.BRAOP) | (1 << MexprParser.OPTOP) | (1 << MexprParser.TEXT))) != 0)):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MatchExprElementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return MexprParser.RULE_matchExprElement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class MatchExprCharsContext(MatchExprElementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MexprParser.MatchExprElementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def TEXT(self):
            return self.getToken(MexprParser.TEXT, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatchExprChars" ):
                listener.enterMatchExprChars(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatchExprChars" ):
                listener.exitMatchExprChars(self)


    class MatchExprOptionalContext(MatchExprElementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MexprParser.MatchExprElementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def OPTOP(self):
            return self.getToken(MexprParser.OPTOP, 0)
        def OPTTXT(self):
            return self.getToken(MexprParser.OPTTXT, 0)
        def OPTCL(self):
            return self.getToken(MexprParser.OPTCL, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatchExprOptional" ):
                listener.enterMatchExprOptional(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatchExprOptional" ):
                listener.exitMatchExprOptional(self)


    class MatchExprVarContext(MatchExprElementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MexprParser.MatchExprElementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def BRAOP(self):
            return self.getToken(MexprParser.BRAOP, 0)
        def varType(self):
            return self.getTypedRuleContext(MexprParser.VarTypeContext,0)

        def ID(self):
            return self.getToken(MexprParser.ID, 0)
        def BRACL(self):
            return self.getToken(MexprParser.BRACL, 0)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatchExprVar" ):
                listener.enterMatchExprVar(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatchExprVar" ):
                listener.exitMatchExprVar(self)


    class MatchExprPlaceholderContext(MatchExprElementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a MexprParser.MatchExprElementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def BRAOP(self):
            return self.getToken(MexprParser.BRAOP, 0)
        def PLACEHOLDER_OP(self):
            return self.getToken(MexprParser.PLACEHOLDER_OP, 0)
        def mexprPlaceholderParam(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(MexprParser.MexprPlaceholderParamContext)
            else:
                return self.getTypedRuleContext(MexprParser.MexprPlaceholderParamContext,i)

        def PLACEHOLDER_CL(self):
            return self.getToken(MexprParser.PLACEHOLDER_CL, 0)
        def BRACL(self):
            return self.getToken(MexprParser.BRACL, 0)
        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(MexprParser.COMMA)
            else:
                return self.getToken(MexprParser.COMMA, i)

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMatchExprPlaceholder" ):
                listener.enterMatchExprPlaceholder(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMatchExprPlaceholder" ):
                listener.exitMatchExprPlaceholder(self)



    def matchExprElement(self):

        localctx = MexprParser.MatchExprElementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_matchExprElement)
        self._la = 0 # Token type
        try:
            self.state = 35
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                localctx = MexprParser.MatchExprVarContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 13
                self.match(MexprParser.BRAOP)
                self.state = 14
                self.varType()
                self.state = 15
                self.match(MexprParser.ID)
                self.state = 16
                self.match(MexprParser.BRACL)
                pass

            elif la_ == 2:
                localctx = MexprParser.MatchExprPlaceholderContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 18
                self.match(MexprParser.BRAOP)
                self.state = 19
                self.match(MexprParser.PLACEHOLDER_OP)
                self.state = 20
                self.mexprPlaceholderParam()
                self.state = 25
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==MexprParser.COMMA:
                    self.state = 21
                    self.match(MexprParser.COMMA)
                    self.state = 22
                    self.mexprPlaceholderParam()
                    self.state = 27
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 28
                self.match(MexprParser.PLACEHOLDER_CL)
                self.state = 29
                self.match(MexprParser.BRACL)
                pass

            elif la_ == 3:
                localctx = MexprParser.MatchExprOptionalContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 31
                self.match(MexprParser.OPTOP)
                self.state = 32
                self.match(MexprParser.OPTTXT)
                self.state = 33
                self.match(MexprParser.OPTCL)
                pass

            elif la_ == 4:
                localctx = MexprParser.MatchExprCharsContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 34
                self.match(MexprParser.TEXT)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MexprPlaceholderParamContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def varType(self):
            return self.getTypedRuleContext(MexprParser.VarTypeContext,0)


        def ID(self):
            return self.getToken(MexprParser.ID, 0)

        def getRuleIndex(self):
            return MexprParser.RULE_mexprPlaceholderParam

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMexprPlaceholderParam" ):
                listener.enterMexprPlaceholderParam(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMexprPlaceholderParam" ):
                listener.exitMexprPlaceholderParam(self)




    def mexprPlaceholderParam(self):

        localctx = MexprParser.MexprPlaceholderParamContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_mexprPlaceholderParam)
        try:
            self.state = 41
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [MexprParser.LT]:
                self.enterOuterAlt(localctx, 1)
                self.state = 37
                self.varType()
                self.state = 38
                self.match(MexprParser.ID)
                pass
            elif token in [MexprParser.ID]:
                self.enterOuterAlt(localctx, 2)
                self.state = 40
                self.match(MexprParser.ID)
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


    class VarTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LT(self):
            return self.getToken(MexprParser.LT, 0)

        def ID(self):
            return self.getToken(MexprParser.ID, 0)

        def GT(self):
            return self.getToken(MexprParser.GT, 0)

        def getRuleIndex(self):
            return MexprParser.RULE_varType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVarType" ):
                listener.enterVarType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVarType" ):
                listener.exitVarType(self)




    def varType(self):

        localctx = MexprParser.VarTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_varType)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 43
            self.match(MexprParser.LT)
            self.state = 44
            self.match(MexprParser.ID)
            self.state = 45
            self.match(MexprParser.GT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





