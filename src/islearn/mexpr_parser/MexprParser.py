# Generated from MexprParser.g4 by ANTLR 4.7.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys

def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3\20")
        buf.write("\62\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\3\2\6\2\f\n\2\r\2")
        buf.write("\16\2\r\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\7\3\32")
        buf.write("\n\3\f\3\16\3\35\13\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3")
        buf.write("&\n\3\3\4\3\4\3\4\3\4\5\4,\n\4\3\5\3\5\3\5\3\5\3\5\2\2")
        buf.write("\6\2\4\6\b\2\2\2\63\2\13\3\2\2\2\4%\3\2\2\2\6+\3\2\2\2")
        buf.write("\b-\3\2\2\2\n\f\5\4\3\2\13\n\3\2\2\2\f\r\3\2\2\2\r\13")
        buf.write("\3\2\2\2\r\16\3\2\2\2\16\3\3\2\2\2\17\20\7\3\2\2\20\21")
        buf.write("\5\b\5\2\21\22\7\13\2\2\22\23\7\7\2\2\23&\3\2\2\2\24\25")
        buf.write("\7\3\2\2\25\26\7\b\2\2\26\33\5\6\4\2\27\30\7\n\2\2\30")
        buf.write("\32\5\6\4\2\31\27\3\2\2\2\32\35\3\2\2\2\33\31\3\2\2\2")
        buf.write("\33\34\3\2\2\2\34\36\3\2\2\2\35\33\3\2\2\2\36\37\7\t\2")
        buf.write("\2\37 \7\7\2\2 &\3\2\2\2!\"\7\4\2\2\"#\7\20\2\2#&\7\17")
        buf.write("\2\2$&\7\5\2\2%\17\3\2\2\2%\24\3\2\2\2%!\3\2\2\2%$\3\2")
        buf.write("\2\2&\5\3\2\2\2\'(\5\b\5\2()\7\13\2\2),\3\2\2\2*,\7\13")
        buf.write("\2\2+\'\3\2\2\2+*\3\2\2\2,\7\3\2\2\2-.\7\r\2\2./\7\13")
        buf.write("\2\2/\60\7\f\2\2\60\t\3\2\2\2\6\r\33%+")
        return buf.getvalue()


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
        self.checkVersion("4.7.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None



    class MatchExprContext(ParserRuleContext):

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





