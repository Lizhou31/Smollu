//===- SmolluAST.h - Smollu AST structures ----------------------*- C++ -*-===//
//
// AST structures and parser interface for Smollu language
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_AST_H
#define SMOLLU_AST_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

extern "C" {
#include "../../compiler/smollu_compiler.h"
}

namespace mlir {
namespace smollu {

// AST node for Smollu parser
struct SmolluASTNode {
    std::string type;
    std::string value;
    std::vector<SmolluASTNode> children;
    int line, column;

    SmolluASTNode(const std::string &t, int l = 0, int c = 0)
        : type(t), line(l), column(c) {}
};

// AST-only parser that generates pure AST without MLIR
class SmolluASTParser {
private:
    // Token management
    Lexer *lexer;
    Token current;
    SmolluASTNode astRoot = SmolluASTNode("Program");

public:
    SmolluASTParser();
    ~SmolluASTParser() = default;

    // Parse source code and return AST
    SmolluASTNode parseSourceFile(const std::string &source);

    // Print AST to output stream
    void printAST(const SmolluASTNode &node, std::ostream &out = std::cout, int depth = 0) const;

private:
    // Token management
    void parserAdvance();
    bool parserCheck(TokenType t);
    bool parserMatch(TokenType t);
    void parserExpected(TokenType t, const char *msg);

    // Parsing methods
    bool parseMainBlock();
    bool parseInitBlock();
    bool parseFunctionsBlock();
    SmolluASTNode parseStatement();
    SmolluASTNode parseAssignment(bool isLocal);
    SmolluASTNode parseExpression();
    SmolluASTNode parseExpressionStatement();

    // Expression parsing with precedence
    SmolluASTNode parseLogicOr();
    SmolluASTNode parseLogicAnd();
    SmolluASTNode parseEquality();
    SmolluASTNode parseComparison();
    SmolluASTNode parseTerm();
    SmolluASTNode parseFactor();
    SmolluASTNode parseUnary();
    SmolluASTNode parsePostfix();
    SmolluASTNode parsePrimary();

    // Control flow
    SmolluASTNode parseWhileStatement();
    SmolluASTNode parseIfStatement();
    SmolluASTNode parseBlock();

    // Native calls
    SmolluASTNode parseNativeCall();
    SmolluASTNode parseNativeCallExpression();

    // Utility
    void printIndent(std::ostream &out, int depth) const;
};

} // namespace smollu
} // namespace mlir

#endif // SMOLLU_AST_H