//===- ASTParser.cpp - Smollu AST-only parser ------------------*- C++ -*-===//
//
// Parser that converts Smollu source code to AST representation only
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluAST.h"
#include <iostream>

using namespace mlir::smollu;

SmolluASTParser::SmolluASTParser() : current() {
}

const char* SmolluASTParser::tokenKindName(::smollu::TokenKind kind) {
    // Map TokenKind enum to readable names
    switch (kind) {
        case ::smollu::TokenKind::Eof: return "EOF";
        case ::smollu::TokenKind::Identifier: return "identifier";
        case ::smollu::TokenKind::KwInit: return "'init'";
        case ::smollu::TokenKind::KwMain: return "'main'";
        case ::smollu::TokenKind::KwFunctions: return "'functions'";
        case ::smollu::TokenKind::KwLocal: return "'local'";
        case ::smollu::TokenKind::KwWhile: return "'while'";
        case ::smollu::TokenKind::KwIf: return "'if'";
        case ::smollu::TokenKind::KwElif: return "'elif'";
        case ::smollu::TokenKind::KwElse: return "'else'";
        case ::smollu::TokenKind::KwFunction: return "'function'";
        case ::smollu::TokenKind::KwReturn: return "'return'";
        case ::smollu::TokenKind::KwNative: return "'native'";
        case ::smollu::TokenKind::LBrace: return "'{'";
        case ::smollu::TokenKind::RBrace: return "'}'";
        case ::smollu::TokenKind::LParen: return "'('";
        case ::smollu::TokenKind::RParen: return "')'";
        case ::smollu::TokenKind::Semicolon: return "';'";
        case ::smollu::TokenKind::Equal: return "'='";
        case ::smollu::TokenKind::Comma: return "','";
        default: return "<unknown>";
    }
}

mlir::smollu::SmolluASTNode SmolluASTParser::parseSourceFile(const std::string &source, const std::string &filename) {
    std::cout << "=== AST Parser: Starting parse (" << filename << ") ===\n";

    currentFilename = filename;

    // Create C++ lexer
    lexer = std::make_unique<::smollu::Lexer>(source, filename);

    // Initialize first token
    parserAdvance();

    // Parse init block
    if (!parseInitBlock()) {
        std::cout << "ERROR: Failed to parse init block\n";
        return SmolluASTNode(""); // Error
    }

    // Parse main block only
    if (!parseMainBlock()) {
        std::cout << "ERROR: Failed to parse main block\n";
        return SmolluASTNode(""); // Error
    }

    // Parse functions block
    if (!parseFunctionsBlock()) {
        std::cout << "ERROR: Failed to parse functions block\n";
        return SmolluASTNode(""); // Error
    }

    return astRoot;
}

void SmolluASTParser::printAST(const SmolluASTNode &node, std::ostream &out, int depth) const {
    printIndent(out, depth);
    out << node.type;
    if (!node.value.empty()) {
        out << " (" << node.value << ")";
    }
    out << "\n";

    for (const auto &child : node.children) {
        printAST(child, out, depth + 1);
    }
}

void SmolluASTParser::parserAdvance() {
    current = lexer->nextToken();
}

bool SmolluASTParser::parserCheck(::smollu::TokenKind t) {
    return current.is(t);
}

bool SmolluASTParser::parserMatch(::smollu::TokenKind t) {
    if (parserCheck(t)) {
        parserAdvance();
        return true;
    }
    return false;
}

void SmolluASTParser::parserExpected(::smollu::TokenKind t, const char *msg) {
    if (!parserMatch(t)) {
        const auto& loc = current.getLocation();
        std::cerr << "[AST Parser] Error at " << loc.line << ":" << loc.column
                  << ": expected " << msg << " (" << tokenKindName(t) << ")\n";
        exit(1);
    }
}

void SmolluASTParser::printIndent(std::ostream &out, int depth) const {
    for (int i = 0; i < depth; i++) out << "  ";
}

bool SmolluASTParser::parseInitBlock() {

    // Skip to "init {"
    while (!parserCheck(::smollu::TokenKind::KwInit) && !parserCheck(::smollu::TokenKind::Eof)) {
        parserAdvance();
    }

    if (parserCheck(::smollu::TokenKind::Eof)) {
        std::cout << "ERROR: Reached EOF while looking for 'init'\n";
        return false;
    }

    parserExpected(::smollu::TokenKind::KwInit, "init");
    parserExpected(::smollu::TokenKind::LBrace, "{");

    // Create AST node for init
    SmolluASTNode initNode = makeNode("InitBlock", current.getLocation().line, current.getLocation().column);

    // Parse statements until '}'
    while (!parserCheck(::smollu::TokenKind::RBrace) && !parserCheck(::smollu::TokenKind::Eof)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        initNode.children.push_back(stmtNode);
    }

    parserExpected(::smollu::TokenKind::RBrace, "}");

    astRoot.children.push_back(initNode);
    return true;
}

bool SmolluASTParser::parseFunctionsBlock() {

    // Skip to "functions {"
    while (!parserCheck(::smollu::TokenKind::KwFunctions) && !parserCheck(::smollu::TokenKind::Eof)) {
        parserAdvance();
    }

    if (parserCheck(::smollu::TokenKind::Eof)) {
        std::cout << "ERROR: Reached EOF while looking for 'functions'\n";
        return false;
    }

    parserExpected(::smollu::TokenKind::KwFunctions, "functions");
    parserExpected(::smollu::TokenKind::LBrace, "{");

    // Create AST node for functions
    SmolluASTNode functionsNode = makeNode("FunctionsBlock", current.getLocation().line, current.getLocation().column);

    // Parse statements until '}'
    while (!parserCheck(::smollu::TokenKind::RBrace) && !parserCheck(::smollu::TokenKind::Eof)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        functionsNode.children.push_back(stmtNode);
    }

    parserExpected(::smollu::TokenKind::RBrace, "}");

    astRoot.children.push_back(functionsNode);
    return true;
}

bool SmolluASTParser::parseMainBlock() {

    // Skip to "main {"
    while (!parserCheck(::smollu::TokenKind::KwMain) && !parserCheck(::smollu::TokenKind::Eof)) {
        parserAdvance();
    }

    if (parserCheck(::smollu::TokenKind::Eof)) {
        std::cout << "ERROR: Reached EOF while looking for 'main'\n";
        return false;
    }

    parserExpected(::smollu::TokenKind::KwMain, "main");
    parserExpected(::smollu::TokenKind::LBrace, "{");

    // Create AST node for main
    SmolluASTNode mainNode = makeNode("MainBlock", current.getLocation().line, current.getLocation().column);

    // Parse statements until '}'
    while (!parserCheck(::smollu::TokenKind::RBrace) && !parserCheck(::smollu::TokenKind::Eof)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        mainNode.children.push_back(stmtNode);
    }

    parserExpected(::smollu::TokenKind::RBrace, "}");

    astRoot.children.push_back(mainNode);
    return true;
}

SmolluASTNode SmolluASTParser::parseStatement() {
    switch (current.getKind()) {
        case ::smollu::TokenKind::Identifier:
            return parseAssignment(false); // Global assignment

        case ::smollu::TokenKind::KwLocal:
            parserAdvance(); // consume 'local'
            return parseAssignment(true); // Local assignment

        case ::smollu::TokenKind::KwWhile:
            return parseWhileStatement();

        case ::smollu::TokenKind::KwIf:
            return parseIfStatement();

        case ::smollu::TokenKind::KwNative:
            return parseNativeCall();

        case ::smollu::TokenKind::KwFunction:
            return parseFunctionDefinition();

        case ::smollu::TokenKind::KwReturn:
            return parseReturnStatement();

        default:
            // Expression statement
            return parseExpressionStatement();
    }
}

SmolluASTNode SmolluASTParser::parseAssignment(bool isLocal) {
    const auto& loc = current.getLocation();
    SmolluASTNode assignNode = makeNode(isLocal ? "LocalAssignment" : "Assignment", loc.line, loc.column);

    if (!parserCheck(::smollu::TokenKind::Identifier)) {
        return SmolluASTNode(""); // Error
    }

    std::string varName(current.getLexeme());
    assignNode.value = varName;
    parserAdvance();

    parserExpected(::smollu::TokenKind::Equal, "=");

    SmolluASTNode exprNode = parseExpression();
    if (exprNode.type.empty()) {
        return SmolluASTNode(""); // Error
    }
    assignNode.children.push_back(exprNode);

    parserExpected(::smollu::TokenKind::Semicolon, ";");

    return assignNode;
}

SmolluASTNode SmolluASTParser::parseExpression() {
    return parseLogicOr();
}

SmolluASTNode SmolluASTParser::parseExpressionStatement() {
    SmolluASTNode exprNode = parseExpression();
    parserExpected(::smollu::TokenKind::Semicolon, ";");
    return exprNode;
}

SmolluASTNode SmolluASTParser::parseLogicOr() {
    SmolluASTNode left = parseLogicAnd();

    while (parserCheck(::smollu::TokenKind::OrOr)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseLogicAnd();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseLogicAnd() {
    SmolluASTNode left = parseEquality();

    while (parserCheck(::smollu::TokenKind::AndAnd)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseEquality();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseEquality() {
    SmolluASTNode left = parseComparison();

    while (parserCheck(::smollu::TokenKind::EqualEqual) || parserCheck(::smollu::TokenKind::BangEqual)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseComparison();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseComparison() {
    SmolluASTNode left = parseTerm();

    while (parserCheck(::smollu::TokenKind::Less) || parserCheck(::smollu::TokenKind::LessEqual) ||
           parserCheck(::smollu::TokenKind::Greater) || parserCheck(::smollu::TokenKind::GreaterEqual)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseTerm();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseTerm() {
    SmolluASTNode left = parseFactor();

    while (parserCheck(::smollu::TokenKind::Plus) || parserCheck(::smollu::TokenKind::Minus)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseFactor();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseFactor() {
    SmolluASTNode left = parseUnary();

    while (parserCheck(::smollu::TokenKind::Star) || parserCheck(::smollu::TokenKind::Slash) || parserCheck(::smollu::TokenKind::Percent)) {
        std::string op(current.getLexeme());
        parserAdvance();
        SmolluASTNode right = parseUnary();

        SmolluASTNode binOp("BinaryOp", left.line, left.column, left.filename);
        binOp.value = op;
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseUnary() {
    if (parserCheck(::smollu::TokenKind::Bang) || parserCheck(::smollu::TokenKind::Minus)) {
        std::string op(current.getLexeme());
        ::smollu::Token tok = current;
        parserAdvance();
        SmolluASTNode operand = parseUnary();

        SmolluASTNode unaryOp("UnaryOp", tok.getLocation().line, tok.getLocation().column, currentFilename);
        unaryOp.value = op;
        unaryOp.children.push_back(operand);
        return unaryOp;
    }
    return parsePostfix();
}

SmolluASTNode SmolluASTParser::parsePostfix() {
    SmolluASTNode expr = parsePrimary();

    while (parserCheck(::smollu::TokenKind::LParen)) {
        if (expr.type != "Identifier") {
            // Error: function call on non-identifier
            return SmolluASTNode("");
        }

        ::smollu::Token tok = current;
        parserAdvance(); // consume '('

        SmolluASTNode funcCall("FunctionCall", tok.getLocation().line, tok.getLocation().column, currentFilename);
        funcCall.value = expr.value;

        // Parse arguments
        if (!parserCheck(::smollu::TokenKind::RParen)) {
            do {
                SmolluASTNode arg = parseExpression();
                if (arg.type.empty()) return SmolluASTNode("");
                funcCall.children.push_back(arg);
            } while (parserMatch(::smollu::TokenKind::Comma));
        }

        parserExpected(::smollu::TokenKind::RParen, ")");
        expr = funcCall;
    }

    return expr;
}

SmolluASTNode SmolluASTParser::parsePrimary() {
    ::smollu::Token tok = current;

    switch (tok.getKind()) {
        case ::smollu::TokenKind::IntLiteral: {
            SmolluASTNode intNode("IntLiteral", tok.getLocation().line, tok.getLocation().column, currentFilename);
            intNode.value = std::to_string(tok.getIntValue());
            parserAdvance();
            return intNode;
        }

        case ::smollu::TokenKind::FloatLiteral: {
            SmolluASTNode floatNode("FloatLiteral", tok.getLocation().line, tok.getLocation().column, currentFilename);
            floatNode.value = std::to_string(tok.getFloatValue());
            parserAdvance();
            return floatNode;
        }

        case ::smollu::TokenKind::BoolLiteral: {
            SmolluASTNode boolNode("BoolLiteral", tok.getLocation().line, tok.getLocation().column, currentFilename);
            boolNode.value = tok.getBoolValue() ? "true" : "false";
            parserAdvance();
            return boolNode;
        }

        case ::smollu::TokenKind::NilLiteral: {
            SmolluASTNode nilNode("NilLiteral", tok.getLocation().line, tok.getLocation().column, currentFilename);
            parserAdvance();
            return nilNode;
        }

        case ::smollu::TokenKind::Identifier: {
            SmolluASTNode identNode("Identifier", tok.getLocation().line, tok.getLocation().column, currentFilename);
            identNode.value = std::string(tok.getLexeme());
            parserAdvance();
            return identNode;
        }

        case ::smollu::TokenKind::KwNative: {
            return parseNativeCallExpression();
        }

        case ::smollu::TokenKind::LParen: {
            parserAdvance(); // consume '('
            SmolluASTNode expr = parseExpression();
            parserExpected(::smollu::TokenKind::RParen, ")");
            return expr;
        }

        default:
            std::cerr << "[AST Parser] Unexpected token " << tokenKindName(tok.getKind())
                      << " at " << tok.getLocation().line << ":" << tok.getLocation().column << "\n";
            return SmolluASTNode(""); // Error
    }
}

SmolluASTNode SmolluASTParser::parseWhileStatement() {
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'while'

    SmolluASTNode whileNode("WhileStatement", tok.getLocation().line, tok.getLocation().column, currentFilename);

    parserExpected(::smollu::TokenKind::LParen, "(");
    SmolluASTNode condition = parseExpression();
    if (condition.type.empty()) return SmolluASTNode("");
    whileNode.children.push_back(condition);

    parserExpected(::smollu::TokenKind::RParen, ")");
    SmolluASTNode body = parseBlock();
    if (body.type.empty()) return SmolluASTNode("");
    whileNode.children.push_back(body);

    return whileNode;
}

SmolluASTNode SmolluASTParser::parseIfStatement() {
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'if'

    SmolluASTNode ifNode("IfStatement", tok.getLocation().line, tok.getLocation().column, currentFilename);

    parserExpected(::smollu::TokenKind::LParen, "(");
    SmolluASTNode condition = parseExpression();
    if (condition.type.empty()) return SmolluASTNode("");
    ifNode.children.push_back(condition);

    parserExpected(::smollu::TokenKind::RParen, ")");
    SmolluASTNode thenBody = parseBlock();
    if (thenBody.type.empty()) return SmolluASTNode("");
    ifNode.children.push_back(thenBody);

    // Check for elif/else
    if (parserCheck(::smollu::TokenKind::KwElif)) {
        SmolluASTNode elifNode = parseIfStatement(); // Recursive for elif
        ifNode.children.push_back(elifNode);
    } else if (parserCheck(::smollu::TokenKind::KwElse)) {
        parserAdvance();
        SmolluASTNode elseBody = parseBlock();
        if (elseBody.type.empty()) return SmolluASTNode("");
        ifNode.children.push_back(elseBody);
    }

    return ifNode;
}

SmolluASTNode SmolluASTParser::parseFunctionDefinition() {
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'function'

    // expect function name
    if (!parserCheck(::smollu::TokenKind::Identifier)) {
        return SmolluASTNode(""); // Error
    }

    SmolluASTNode functionNode("FunctionDefinition", tok.getLocation().line, tok.getLocation().column, currentFilename);
    functionNode.value = std::string(current.getLexeme());
    parserAdvance(); // consume function name

    // expect '('
    parserExpected(::smollu::TokenKind::LParen, "(");

    // Parse arguments
    if (!parserCheck(::smollu::TokenKind::RParen)) {
        do {
            SmolluASTNode arg = parseExpression();
            if (arg.type.empty()) return SmolluASTNode("");
            functionNode.children.push_back(arg);
        } while (parserMatch(::smollu::TokenKind::Comma));
    }

    // expect ')'
    parserExpected(::smollu::TokenKind::RParen, ")");

    // parse function body
    SmolluASTNode body = parseBlock();
    if (body.type.empty()) return SmolluASTNode("");
    functionNode.children.push_back(body);

    return functionNode;
}

SmolluASTNode SmolluASTParser::parseReturnStatement() {
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'return'

    SmolluASTNode returnNode("ReturnStatement", tok.getLocation().line, tok.getLocation().column, currentFilename);

    SmolluASTNode exprNode = parseExpression();
    if (exprNode.type.empty()) return SmolluASTNode("");
    returnNode.children.push_back(exprNode);

    parserExpected(::smollu::TokenKind::Semicolon, ";");

    return returnNode;
}

SmolluASTNode SmolluASTParser::parseBlock() {
    ::smollu::Token tok = current;
    parserExpected(::smollu::TokenKind::LBrace, "{");

    SmolluASTNode blockNode("Block", tok.getLocation().line, tok.getLocation().column, currentFilename);

    while (!parserCheck(::smollu::TokenKind::RBrace) && !parserCheck(::smollu::TokenKind::Eof)) {
        SmolluASTNode stmt = parseStatement();
        if (stmt.type.empty()) return SmolluASTNode("");
        blockNode.children.push_back(stmt);
    }

    parserExpected(::smollu::TokenKind::RBrace, "}");
    return blockNode;
}

SmolluASTNode SmolluASTParser::parseNativeCall() {
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'native'

    if (!parserCheck(::smollu::TokenKind::Identifier)) {
        return SmolluASTNode(""); // Error
    }

    SmolluASTNode nativeNode("NativeCall", tok.getLocation().line, tok.getLocation().column, currentFilename);
    nativeNode.value = std::string(current.getLexeme());
    parserAdvance();

    parserExpected(::smollu::TokenKind::LParen, "(");

    // Parse arguments
    if (!parserCheck(::smollu::TokenKind::RParen)) {
        do {
            SmolluASTNode arg = parseExpression();
            if (arg.type.empty()) return SmolluASTNode("");
            nativeNode.children.push_back(arg);
        } while (parserMatch(::smollu::TokenKind::Comma));
    }

    parserExpected(::smollu::TokenKind::RParen, ")");
    parserExpected(::smollu::TokenKind::Semicolon, ";");

    return nativeNode;
}

SmolluASTNode SmolluASTParser::parseNativeCallExpression() {
    // Same as parseNativeCall but without semicolon for expression context
    ::smollu::Token tok = current;
    parserAdvance(); // consume 'native'

    if (!parserCheck(::smollu::TokenKind::Identifier)) {
        return SmolluASTNode(""); // Error
    }

    SmolluASTNode nativeNode("NativeCall", tok.getLocation().line, tok.getLocation().column, currentFilename);
    nativeNode.value = std::string(current.getLexeme());
    parserAdvance();

    parserExpected(::smollu::TokenKind::LParen, "(");

    // Parse arguments
    if (!parserCheck(::smollu::TokenKind::RParen)) {
        do {
            SmolluASTNode arg = parseExpression();
            if (arg.type.empty()) return SmolluASTNode("");
            nativeNode.children.push_back(arg);
        } while (parserMatch(::smollu::TokenKind::Comma));
    }

    parserExpected(::smollu::TokenKind::RParen, ")");
    return nativeNode;
}