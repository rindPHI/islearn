parser grammar MexprParser;

options { tokenVocab=MexprLexer; }

matchExpr: matchExprElement + ;

matchExprElement:
    BRAOP varType ID BRACL  # MatchExprVar
  | BRAOP PLACEHOLDER_OP mexprPlaceholderParam (COMMA mexprPlaceholderParam) * PLACEHOLDER_CL BRACL # MatchExprPlaceholder
  | OPTOP OPTTXT OPTCL      # MatchExprOptional
  | TEXT                    # MatchExprChars
  ;

mexprPlaceholderParam:
    varType ID
  | ID
  ;

varType : LT ID GT ;
