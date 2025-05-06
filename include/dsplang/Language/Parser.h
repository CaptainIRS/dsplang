// Adapted from LLVM MLIR Toy example

#ifndef DSPLANG_PARSER_H
#define DSPLANG_PARSER_H

#include "dsplang/Language/AST.h"
#include "dsplang/Language/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace dsplang {

/// This is a simple recursive parser for the Dsplang language. It produces a
/// well formed AST from a stream of Token supplied by the Lexer. No semantic
/// checks or symbol resolution is performed. For example, variables are
/// referenced by string and the code could reference an undeclared variable and
/// the parsing succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in this vector.
    std::vector<FunctionAST> functions;
    while (auto f = parseDefinition()) {
      functions.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    // return takes an optional argument
    std::optional<std::unique_ptr<ExprAST>> expr;
    if (lexer.getCurToken() != ';') {
      expr = parseExpression();
      if (!expr)
        return nullptr;
    }
    return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result =
        std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(result);
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]

    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());

    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
          return llvm::isa<LiteralExprAST>(expr.get());
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      auto firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }
    return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                            std::move(dims));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto v = parseExpression();
    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return v;
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string name(lexer.getId());

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    if (lexer.getCurToken() != '(') // Simple variable ref.
      return std::make_unique<VariableExprAST>(std::move(loc), name);

    // This is a function call.
    lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto arg = parseExpression())
          args.push_back(std::move(arg));
        else
          return nullptr;

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (name == "print") {
      if (args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
    }

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case '(':
      return parseParenExpr();
    case '[':
      return parseTensorLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int tokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrec < exprPrec)
        return lhs;

      // Okay, we know this is a binop.
      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                            std::move(lhs), std::move(rhs));
    }
  }

  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;

    return parseBinOpRHS(0, std::move(lhs));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num , element_type
  /// element_type ::= si8 | ui8 | si16 | ui16 | si32 | ui32 | f16 | f32
  std::unique_ptr<VarType> parseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    lexer.getNextToken(); // eat <

    auto type = std::make_unique<VarType>();
    if (lexer.getCurToken() != tok_number)
      return parseError<VarType>("number", "to begin type");
    type->shape = lexer.getValue();
    lexer.getNextToken();
    if (lexer.getCurToken() == ',')
      lexer.getNextToken();
    else
      return parseError<VarType>(",", "to continue type");

    if (lexer.getCurToken() != tok_type_si8 &&
        lexer.getCurToken() != tok_type_ui8 &&
        lexer.getCurToken() != tok_type_si16 &&
        lexer.getCurToken() != tok_type_ui16 &&
        lexer.getCurToken() != tok_type_si32 &&
        lexer.getCurToken() != tok_type_ui32 &&
        lexer.getCurToken() != tok_type_f16 &&
        lexer.getCurToken() != tok_type_f32)
      return parseError<VarType>("type", "to end type");
    switch (lexer.getCurToken()) {
    case tok_type_si8:
      type->type = VarType::Type::si8;
      break;
    case tok_type_ui8:
      type->type = VarType::Type::ui8;
      break;
    case tok_type_si16:
      type->type = VarType::Type::si16;
      break;
    case tok_type_ui16:
      type->type = VarType::Type::ui16;
      break;
    case tok_type_si32:
      type->type = VarType::Type::si32;
      break;
    case tok_type_ui32:
      type->type = VarType::Type::ui32;
      break;
    case tok_type_f16:
      type->type = VarType::Type::f16;
      break;
    case tok_type_f32:
      type->type = VarType::Type::f32;
      break;
    default:
      return parseError<VarType>("type", "to end type");
    }
    lexer.getNextToken(); // eat type

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    lexer.getNextToken(); // eat >
    return type;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= var identifier [ type ] = expr
  std::unique_ptr<VarDeclExprAST> parseDeclaration() {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat var

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identified",
                                        "after 'var' declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    std::unique_ptr<VarType> type; // Type is not optional
    if (lexer.getCurToken() == '<') {
      type = parseType();
      if (!type)
        return nullptr;
    } else {
      return parseError<VarDeclExprAST>(
          "<", "to begin type after variable declaration");
    }

    if (!type)
      type = std::make_unique<VarType>();
    lexer.consume(Token('='));
    auto expr = parseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                            std::move(*type), std::move(expr));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> parseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto varDecl = parseDeclaration();
        if (!varDecl)
          return nullptr;
        exprList->push_back(std::move(varDecl));
      } else if (lexer.getCurToken() == tok_return) {
        // Return statement
        auto ret = parseReturn();
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else {
        // General expression
        auto expr = parseExpression();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return exprList;
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_def)
      return parseError<PrototypeAST>("def", "in prototype");
    lexer.consume(tok_def);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string fnName(lexer.getId());
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<ArgExprAST>> args;
    if (lexer.getCurToken() != ')') {
      do {
        std::string name(lexer.getId());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        auto type = std::make_unique<VarType>();
        bool isPtr = false;
        if (lexer.getCurToken() == '*') {
          isPtr = true;
          lexer.consume(Token('*'));
        }
        if (lexer.getCurToken() == '<') {
          type = parseType();
          if (!type)
            return parseError<PrototypeAST>(
                "<", "to begin type after function argument");
        } else {
          return parseError<PrototypeAST>(
              "<", "to begin type after function argument");
        }
        auto arg = std::make_unique<ArgExprAST>(std::move(loc), name,
                                                std::move(*type), isPtr);
        args.push_back(std::move(arg));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    // success.
    lexer.consume(Token(')'));
    return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                          std::move(args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition() {
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    if (auto block = parseBlock())
      return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
    return nullptr;
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '=':
      return 10;
    case '|':
      return 14;
    case '^':
      return 15;
    case '&':
      return 16;
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace dsplang

#endif // DSPLANG_PARSER_H
