//===- ASTParser.cpp - Smollu AST-only parser ------------------*- C++ -*-===//
//
// Parser that converts Smollu source code to AST representation only
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluAST.h"
#include <iostream>

using namespace mlir::smollu;

SmolluASTParser::SmolluASTParser() : current({TOK_EOF, nullptr, 0, 0, {0}}) {
}

mlir::smollu::SmolluASTNode SmolluASTParser::parseSourceFile(const std::string &source) {
    std::cout << "=== AST Parser: Starting parse ===\n";

    Lexer lex;
    lexer_init(&lex, source.c_str());
    lexer = &lex;

    // Initialize first token
    parserAdvance();

    // Parse main block only
    if (!parseMainBlock()) {
        std::cout << "ERROR: Failed to parse main block\n";
        lexer_free(&lex);
        return SmolluASTNode(""); // Error
    }

    lexer_free(&lex);
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
    token_free(&current);
    current = lexer_next(lexer);
}

bool SmolluASTParser::parserCheck(TokenType t) {
    return current.type == t;
}

bool SmolluASTParser::parserMatch(TokenType t) {
    if (parserCheck(t)) {
        parserAdvance();
        return true;
    }
    return false;
}

void SmolluASTParser::parserExpected(TokenType t, const char *msg) {
    if (!parserMatch(t)) {
        std::cerr << "[AST Parser] Error at " << current.line << ":" << current.column
                  << ": expected " << msg << " (" << token_type_name(t) << ")\n";
        exit(1);
    }
}

void SmolluASTParser::printIndent(std::ostream &out, int depth) const {
    for (int i = 0; i < depth; i++) out << "  ";
}

bool SmolluASTParser::parseInitBlock() {

    // Skip to "init {"
    while (!parserCheck(TOK_KW_INIT) && !parserCheck(TOK_EOF)) {
        parserAdvance();
    }

    if (parserCheck(TOK_EOF)) {
        std::cout << "ERROR: Reached EOF while looking for 'init'\n";
        return false;
    }

    parserExpected(TOK_KW_INIT, "init");
    parserExpected(TOK_LBRACE, "{");

    // Create AST node for init
    SmolluASTNode initNode("InitBlock", current.line, current.column);

    // Parse statements until '}'
    while (!parserCheck(TOK_RBRACE) && !parserCheck(TOK_EOF)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        initNode.children.push_back(stmtNode);
    }

    parserExpected(TOK_RBRACE, "}");

    astRoot.children.push_back(initNode);
    return true;
}

bool SmolluASTParser::parseFunctionsBlock() {

    // Skip to "functions {"
    while (!parserCheck(TOK_KW_FUNCTIONS) && !parserCheck(TOK_EOF)) {
        parserAdvance();
    }

    if (parserCheck(TOK_EOF)) {
        std::cout << "ERROR: Reached EOF while looking for 'functions'\n";
        return false;
    }

    parserExpected(TOK_KW_FUNCTIONS, "functions");
    parserExpected(TOK_LBRACE, "{");

    // Create AST node for functions
    SmolluASTNode functionsNode("FunctionsBlock", current.line, current.column);

    // Parse statements until '}'
    while (!parserCheck(TOK_RBRACE) && !parserCheck(TOK_EOF)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        functionsNode.children.push_back(stmtNode);
    }

    parserExpected(TOK_RBRACE, "}");

    astRoot.children.push_back(functionsNode);
    return true;
}

bool SmolluASTParser::parseMainBlock() {

    // Skip to "main {"
    while (!parserCheck(TOK_KW_MAIN) && !parserCheck(TOK_EOF)) {
        parserAdvance();
    }

    if (parserCheck(TOK_EOF)) {
        std::cout << "ERROR: Reached EOF while looking for 'main'\n";
        return false;
    }

    parserExpected(TOK_KW_MAIN, "main");
    parserExpected(TOK_LBRACE, "{");

    // Create AST node for main
    SmolluASTNode mainNode("MainBlock", current.line, current.column);

    // Parse statements until '}'
    while (!parserCheck(TOK_RBRACE) && !parserCheck(TOK_EOF)) {
        SmolluASTNode stmtNode = parseStatement();
        if (stmtNode.type.empty()) {
            return false; // Parse error
        }
        mainNode.children.push_back(stmtNode);
    }

    parserExpected(TOK_RBRACE, "}");

    astRoot.children.push_back(mainNode);
    return true;
}

SmolluASTNode SmolluASTParser::parseStatement() {
    switch (current.type) {
        case TOK_IDENTIFIER:
            return parseAssignment(false); // Global assignment

        case TOK_KW_LOCAL:
            parserAdvance(); // consume 'local'
            return parseAssignment(true); // Local assignment

        case TOK_KW_WHILE:
            return parseWhileStatement();

        case TOK_KW_IF:
            return parseIfStatement();

        case TOK_KW_NATIVE:
            return parseNativeCall();

        default:
            // Expression statement
            return parseExpressionStatement();
    }
}

SmolluASTNode SmolluASTParser::parseAssignment(bool isLocal) {
    SmolluASTNode assignNode(isLocal ? "LocalAssignment" : "Assignment", current.line, current.column);

    if (!parserCheck(TOK_IDENTIFIER)) {
        return SmolluASTNode(""); // Error
    }

    std::string varName(current.lexeme);
    assignNode.value = varName;
    parserAdvance();

    parserExpected(TOK_EQUAL, "=");

    SmolluASTNode exprNode = parseExpression();
    if (exprNode.type.empty()) {
        return SmolluASTNode(""); // Error
    }
    assignNode.children.push_back(exprNode);

    parserExpected(TOK_SEMICOLON, ";");

    return assignNode;
}

SmolluASTNode SmolluASTParser::parseExpression() {
    return parseLogicOr();
}

SmolluASTNode SmolluASTParser::parseExpressionStatement() {
    SmolluASTNode exprNode = parseExpression();
    parserExpected(TOK_SEMICOLON, ";");
    return exprNode;
}

SmolluASTNode SmolluASTParser::parseLogicOr() {
    SmolluASTNode left = parseLogicAnd();

    while (parserCheck(TOK_OR_OR)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseLogicAnd();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseLogicAnd() {
    SmolluASTNode left = parseEquality();

    while (parserCheck(TOK_AND_AND)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseEquality();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseEquality() {
    SmolluASTNode left = parseComparison();

    while (parserCheck(TOK_EQUAL_EQUAL) || parserCheck(TOK_BANG_EQUAL)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseComparison();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseComparison() {
    SmolluASTNode left = parseTerm();

    while (parserCheck(TOK_LESS) || parserCheck(TOK_LESS_EQUAL) ||
           parserCheck(TOK_GREATER) || parserCheck(TOK_GREATER_EQUAL)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseTerm();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseTerm() {
    SmolluASTNode left = parseFactor();

    while (parserCheck(TOK_PLUS) || parserCheck(TOK_MINUS)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseFactor();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseFactor() {
    SmolluASTNode left = parseUnary();

    while (parserCheck(TOK_STAR) || parserCheck(TOK_SLASH) || parserCheck(TOK_PERCENT)) {
        TokenType op = current.type;
        parserAdvance();
        SmolluASTNode right = parseUnary();

        SmolluASTNode binOp("BinaryOp", left.line, left.column);
        binOp.value = token_type_name(op);
        binOp.children.push_back(left);
        binOp.children.push_back(right);
        left = binOp;
    }
    return left;
}

SmolluASTNode SmolluASTParser::parseUnary() {
    if (parserCheck(TOK_BANG) || parserCheck(TOK_MINUS)) {
        TokenType op = current.type;
        Token tok = current;
        parserAdvance();
        SmolluASTNode operand = parseUnary();

        SmolluASTNode unaryOp("UnaryOp", tok.line, tok.column);
        unaryOp.value = token_type_name(op);
        unaryOp.children.push_back(operand);
        return unaryOp;
    }
    return parsePostfix();
}

SmolluASTNode SmolluASTParser::parsePostfix() {
    SmolluASTNode expr = parsePrimary();

    while (parserCheck(TOK_LPAREN)) {
        if (expr.type != "Identifier") {
            // Error: function call on non-identifier
            return SmolluASTNode("");
        }

        Token tok = current;
        parserAdvance(); // consume '('

        SmolluASTNode funcCall("FunctionCall", tok.line, tok.column);
        funcCall.value = expr.value;

        // Parse arguments
        if (!parserCheck(TOK_RPAREN)) {
            do {
                SmolluASTNode arg = parseExpression();
                if (arg.type.empty()) return SmolluASTNode("");
                funcCall.children.push_back(arg);
            } while (parserMatch(TOK_COMMA));
        }

        parserExpected(TOK_RPAREN, ")");
        expr = funcCall;
    }

    return expr;
}

SmolluASTNode SmolluASTParser::parsePrimary() {
    Token tok = current;

    switch (tok.type) {
        case TOK_INT_LITERAL: {
            SmolluASTNode intNode("IntLiteral", tok.line, tok.column);
            intNode.value = std::to_string(tok.value.int_val);
            parserAdvance();
            return intNode;
        }

        case TOK_FLOAT_LITERAL: {
            SmolluASTNode floatNode("FloatLiteral", tok.line, tok.column);
            floatNode.value = std::to_string(tok.value.float_val);
            parserAdvance();
            return floatNode;
        }

        case TOK_BOOL_LITERAL: {
            SmolluASTNode boolNode("BoolLiteral", tok.line, tok.column);
            boolNode.value = tok.value.bool_val ? "true" : "false";
            parserAdvance();
            return boolNode;
        }

        case TOK_NIL_LITERAL: {
            SmolluASTNode nilNode("NilLiteral", tok.line, tok.column);
            parserAdvance();
            return nilNode;
        }

        case TOK_IDENTIFIER: {
            SmolluASTNode identNode("Identifier", tok.line, tok.column);
            identNode.value = std::string(tok.lexeme);
            parserAdvance();
            return identNode;
        }

        case TOK_KW_NATIVE: {
            return parseNativeCallExpression();
        }

        case TOK_LPAREN: {
            parserAdvance(); // consume '('
            SmolluASTNode expr = parseExpression();
            parserExpected(TOK_RPAREN, ")");
            return expr;
        }

        default:
            std::cerr << "[AST Parser] Unexpected token " << token_type_name(tok.type)
                      << " at " << tok.line << ":" << tok.column << "\n";
            return SmolluASTNode(""); // Error
    }
}

SmolluASTNode SmolluASTParser::parseWhileStatement() {
    Token tok = current;
    parserAdvance(); // consume 'while'

    SmolluASTNode whileNode("WhileStatement", tok.line, tok.column);

    parserExpected(TOK_LPAREN, "(");
    SmolluASTNode condition = parseExpression();
    if (condition.type.empty()) return SmolluASTNode("");
    whileNode.children.push_back(condition);

    parserExpected(TOK_RPAREN, ")");
    SmolluASTNode body = parseBlock();
    if (body.type.empty()) return SmolluASTNode("");
    whileNode.children.push_back(body);

    return whileNode;
}

SmolluASTNode SmolluASTParser::parseIfStatement() {
    Token tok = current;
    parserAdvance(); // consume 'if'

    SmolluASTNode ifNode("IfStatement", tok.line, tok.column);

    parserExpected(TOK_LPAREN, "(");
    SmolluASTNode condition = parseExpression();
    if (condition.type.empty()) return SmolluASTNode("");
    ifNode.children.push_back(condition);

    parserExpected(TOK_RPAREN, ")");
    SmolluASTNode thenBody = parseBlock();
    if (thenBody.type.empty()) return SmolluASTNode("");
    ifNode.children.push_back(thenBody);

    // Check for elif/else
    if (parserCheck(TOK_KW_ELIF)) {
        SmolluASTNode elifNode = parseIfStatement(); // Recursive for elif
        ifNode.children.push_back(elifNode);
    } else if (parserCheck(TOK_KW_ELSE)) {
        parserAdvance();
        SmolluASTNode elseBody = parseBlock();
        if (elseBody.type.empty()) return SmolluASTNode("");
        ifNode.children.push_back(elseBody);
    }

    return ifNode;
}

SmolluASTNode SmolluASTParser::parseBlock() {
    Token tok = current;
    parserExpected(TOK_LBRACE, "{");

    SmolluASTNode blockNode("Block", tok.line, tok.column);

    while (!parserCheck(TOK_RBRACE) && !parserCheck(TOK_EOF)) {
        SmolluASTNode stmt = parseStatement();
        if (stmt.type.empty()) return SmolluASTNode("");
        blockNode.children.push_back(stmt);
    }

    parserExpected(TOK_RBRACE, "}");
    return blockNode;
}

SmolluASTNode SmolluASTParser::parseNativeCall() {
    Token tok = current;
    parserAdvance(); // consume 'native'

    if (!parserCheck(TOK_IDENTIFIER)) {
        return SmolluASTNode(""); // Error
    }

    SmolluASTNode nativeNode("NativeCall", tok.line, tok.column);
    nativeNode.value = std::string(current.lexeme);
    parserAdvance();

    parserExpected(TOK_LPAREN, "(");

    // Parse arguments
    if (!parserCheck(TOK_RPAREN)) {
        do {
            SmolluASTNode arg = parseExpression();
            if (arg.type.empty()) return SmolluASTNode("");
            nativeNode.children.push_back(arg);
        } while (parserMatch(TOK_COMMA));
    }

    parserExpected(TOK_RPAREN, ")");
    parserExpected(TOK_SEMICOLON, ";");

    return nativeNode;
}

SmolluASTNode SmolluASTParser::parseNativeCallExpression() {
    // Same as parseNativeCall but without semicolon for expression context
    Token tok = current;
    parserAdvance(); // consume 'native'

    if (!parserCheck(TOK_IDENTIFIER)) {
        return SmolluASTNode(""); // Error
    }

    SmolluASTNode nativeNode("NativeCall", tok.line, tok.column);
    nativeNode.value = std::string(current.lexeme);
    parserAdvance();

    parserExpected(TOK_LPAREN, "(");

    // Parse arguments
    if (!parserCheck(TOK_RPAREN)) {
        do {
            SmolluASTNode arg = parseExpression();
            if (arg.type.empty()) return SmolluASTNode("");
            nativeNode.children.push_back(arg);
        } while (parserMatch(TOK_COMMA));
    }

    parserExpected(TOK_RPAREN, ")");
    return nativeNode;
}