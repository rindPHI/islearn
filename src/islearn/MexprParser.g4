parser grammar MexprParser;

options { tokenVocab=MexprLexer; }

matchExpr: matchExprElement + ;

matchExprElement:
    BRAOP varType ID BRACL                                    # MatchExprVar
  | BRAOP PLACEHOLDER_OP ID (COMMA ID) * PLACEHOLDER_CL BRACL # MatchExprPlaceholder
  | OPTOP OPTTXT OPTCL                                        # MatchExprOptional
  | TEXT                                                      # MatchExprChars
  ;

varType : LT ID GT ;
