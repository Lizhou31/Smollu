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
#include <memory>
#include "Smollu/Lexer.h"

namespace mlir {
namespace smollu {

// AST node for Smollu parser
struct SmolluASTNode {
    std::string type;
    std::string value;
    std::vector<SmolluASTNode> children;
    std::string filename;  // Source filename for location tracking
    int line, column;

    SmolluASTNode(const std::string &t, int l = 0, int c = 0, const std::string &f = "<unknown>")
        : type(t), filename(f), line(l), column(c) {}
};

// AST-only parser that generates pure AST without MLIR
class SmolluASTParser {
private:
    // Token management
    std::unique_ptr<::smollu::Lexer> lexer;
    ::smollu::Token current;
    SmolluASTNode astRoot = SmolluASTNode("Program");
    std::string currentFilename;  // Current source filename for location tracking

public:
    SmolluASTParser();
    ~SmolluASTParser() = default;

    // Parse source code and return AST
    SmolluASTNode parseSourceFile(const std::string &source, const std::string &filename = "<stdin>");

    // Print AST to output stream
    void printAST(const SmolluASTNode &node, std::ostream &out = std::cout, int depth = 0) const;

private:
    // Token management
    void parserAdvance();
    bool parserCheck(::smollu::TokenKind t);
    bool parserMatch(::smollu::TokenKind t);
    void parserExpected(::smollu::TokenKind t, const char *msg);

    // Parsing methods
    bool parseMainBlock();
    bool parseInitBlock();
    bool parseFunctionsBlock();
    SmolluASTNode parseStatement();
    SmolluASTNode parseAssignment(bool isLocal);
    SmolluASTNode parseExpression();
    SmolluASTNode parseExpressionStatement();
    SmolluASTNode parseReturnStatement();

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

    // Function definitions
    SmolluASTNode parseFunctionDefinition();

    // Utility
    void printIndent(std::ostream &out, int depth) const;

    // Helper to create AST node with current filename
    SmolluASTNode makeNode(const std::string &type, int line, int column) const {
        return SmolluASTNode(type, line, column, currentFilename);
    }
    SmolluASTNode makeNode(const std::string &type) const {
        const auto& loc = current.getLocation();
        return SmolluASTNode(type, loc.line, loc.column, currentFilename);
    }

    // Convert C++ TokenKind to string for error messages
    static const char* tokenKindName(::smollu::TokenKind kind);
};

} // namespace smollu
} // namespace mlir

#endif // SMOLLU_AST_H