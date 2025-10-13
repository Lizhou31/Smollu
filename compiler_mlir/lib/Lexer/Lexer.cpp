/**
 * @file Lexer.cpp
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Implementation of C++ Lexer for Smollu
 *
 * @version 0.2
 * @date 2025-10-03
 *
 * @copyright Copyright (c) 2025 Lizhou
 *
 */

#include "Smollu/Lexer.h"
#include <cctype>
#include <sstream>
#include <unordered_map>

namespace smollu {

/* ──────────────────────────────────────────────────────────────────────────── */
/*  SourceLocation Implementation                                              */
/* ──────────────────────────────────────────────────────────────────────────── */

std::string SourceLocation::toString() const {
    std::ostringstream oss;
    oss << filename << ":" << line << ":" << column;
    return oss.str();
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Token Implementation                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

std::string_view Token::getKindName() const {
    switch (kind_) {
        case TokenKind::Eof:           return "EOF";
        case TokenKind::Unknown:       return "UNKNOWN";
        case TokenKind::Identifier:    return "IDENTIFIER";
        case TokenKind::IntLiteral:    return "INT_LITERAL";
        case TokenKind::FloatLiteral:  return "FLOAT_LITERAL";
        case TokenKind::BoolLiteral:   return "BOOL_LITERAL";
        case TokenKind::NilLiteral:    return "NIL_LITERAL";

        case TokenKind::KwNative:      return "KW_NATIVE";
        case TokenKind::KwFunction:    return "KW_FUNCTION";
        case TokenKind::KwFunctions:   return "KW_FUNCTIONS";
        case TokenKind::KwReturn:      return "KW_RETURN";
        case TokenKind::KwInit:        return "KW_INIT";
        case TokenKind::KwMain:        return "KW_MAIN";
        case TokenKind::KwLocal:       return "KW_LOCAL";
        case TokenKind::KwWhile:       return "KW_WHILE";
        case TokenKind::KwIf:          return "KW_IF";
        case TokenKind::KwElif:        return "KW_ELIF";
        case TokenKind::KwElse:        return "KW_ELSE";

        case TokenKind::Plus:          return "+";
        case TokenKind::Minus:         return "-";
        case TokenKind::Star:          return "*";
        case TokenKind::Slash:         return "/";
        case TokenKind::Percent:       return "%";
        case TokenKind::Equal:         return "=";
        case TokenKind::EqualEqual:    return "==";
        case TokenKind::Bang:          return "!";
        case TokenKind::BangEqual:     return "!=";
        case TokenKind::Greater:       return ">";
        case TokenKind::GreaterEqual:  return ">=";
        case TokenKind::Less:          return "<";
        case TokenKind::LessEqual:     return "<=";
        case TokenKind::AndAnd:        return "&&";
        case TokenKind::OrOr:          return "||";

        case TokenKind::LParen:        return "(";
        case TokenKind::RParen:        return ")";
        case TokenKind::LBrace:        return "{";
        case TokenKind::RBrace:        return "}";
        case TokenKind::Comma:         return ",";
        case TokenKind::Semicolon:     return ";";
    }
    return "<UNKNOWN>";
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Lexer Implementation                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

Lexer::Lexer(std::string source, std::string filename)
    : source_(std::move(source)),
      filename_(std::move(filename)),
      position_(0),
      line_(1),
      column_(1) {}

char Lexer::peek() const {
    return isAtEnd() ? '\0' : source_[position_];
}

char Lexer::peekAhead(size_t n) const {
    size_t pos = position_ + n;
    return pos >= source_.size() ? '\0' : source_[pos];
}

char Lexer::advance() {
    if (isAtEnd()) return '\0';

    char c = source_[position_++];
    if (c == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    return c;
}

bool Lexer::match(char expected) {
    if (peek() != expected) return false;
    advance();
    return true;
}

bool Lexer::isAtEnd() const {
    return position_ >= source_.size();
}

SourceLocation Lexer::getCurrentLocation() const {
    return SourceLocation(filename_, line_, column_, position_);
}

SourceLocation Lexer::makeLocation(size_t offset) const {
    // Calculate line/column for the given offset
    // This is approximate - for exact location, we'd need to track from the offset
    // For simplicity, we use current location adjusted by the difference
    size_t diff = position_ - offset;
    unsigned col = column_ > diff ? column_ - diff : 1;
    return SourceLocation(filename_, line_, col, offset);
}

void Lexer::skipWhitespaceAndComments() {
    while (true) {
        char c = peek();

        // Whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            advance();
            continue;
        }

        // Line comment: --
        if (c == '-' && peekAhead(1) == '-') {
            // Skip until newline or EOF
            while (peek() != '\n' && !isAtEnd()) {
                advance();
            }
            continue;
        }

        break;
    }
}

bool Lexer::isIdentifierStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool Lexer::isIdentifierPart(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

TokenKind Lexer::keywordKind(std::string_view lexeme) {
    static const std::unordered_map<std::string_view, TokenKind> keywords = {
        {"native",    TokenKind::KwNative},
        {"function",  TokenKind::KwFunction},
        {"functions", TokenKind::KwFunctions},
        {"return",    TokenKind::KwReturn},
        {"init",      TokenKind::KwInit},
        {"main",      TokenKind::KwMain},
        {"local",     TokenKind::KwLocal},
        {"while",     TokenKind::KwWhile},
        {"if",        TokenKind::KwIf},
        {"elif",      TokenKind::KwElif},
        {"else",      TokenKind::KwElse},
    };

    auto it = keywords.find(lexeme);
    return it != keywords.end() ? it->second : TokenKind::Identifier;
}

Token Lexer::makeToken(TokenKind kind, size_t start) {
    auto loc = makeLocation(start);
    auto lexeme = std::string_view(source_).substr(start, position_ - start);
    return Token(kind, loc, lexeme);
}

Token Lexer::makeToken(TokenKind kind, size_t start, Token::ValueType value) {
    auto loc = makeLocation(start);
    auto lexeme = std::string_view(source_).substr(start, position_ - start);
    return Token(kind, loc, lexeme, value);
}

Token Lexer::scanIdentifierOrKeyword() {
    size_t start = position_;
    advance(); // Consume first character

    while (isIdentifierPart(peek())) {
        advance();
    }

    auto lexeme = std::string_view(source_).substr(start, position_ - start);

    // Check for boolean literals
    if (lexeme == "true") {
        return makeToken(TokenKind::BoolLiteral, start, true);
    }
    if (lexeme == "false") {
        return makeToken(TokenKind::BoolLiteral, start, false);
    }

    // Check for nil literal
    if (lexeme == "nil") {
        return makeToken(TokenKind::NilLiteral, start);
    }

    // Check if it's a keyword
    TokenKind kind = keywordKind(lexeme);
    return makeToken(kind, start);
}

Token Lexer::scanNumber() {
    size_t start = position_;

    // Handle negative sign
    bool hasSign = peek() == '-';
    if (hasSign) {
        advance();
    }

    // Scan integer part
    while (std::isdigit(static_cast<unsigned char>(peek()))) {
        advance();
    }

    // Check for decimal point
    bool isFloat = false;
    if (peek() == '.' && std::isdigit(static_cast<unsigned char>(peekAhead(1)))) {
        isFloat = true;
        advance(); // Consume '.'

        // Scan fractional part
        while (std::isdigit(static_cast<unsigned char>(peek()))) {
            advance();
        }
    }

    auto lexeme = std::string_view(source_).substr(start, position_ - start);

    if (isFloat) {
        float value = std::stof(std::string(lexeme));
        return makeToken(TokenKind::FloatLiteral, start, value);
    } else {
        int value = std::stoi(std::string(lexeme));
        return makeToken(TokenKind::IntLiteral, start, value);
    }
}

Token Lexer::scanOperatorOrPunct() {
    size_t start = position_;
    char c = advance();

    switch (c) {
        case '+': return makeToken(TokenKind::Plus, start);
        case '-': return makeToken(TokenKind::Minus, start);
        case '*': return makeToken(TokenKind::Star, start);
        case '/': return makeToken(TokenKind::Slash, start);
        case '%': return makeToken(TokenKind::Percent, start);

        case '(': return makeToken(TokenKind::LParen, start);
        case ')': return makeToken(TokenKind::RParen, start);
        case '{': return makeToken(TokenKind::LBrace, start);
        case '}': return makeToken(TokenKind::RBrace, start);
        case ',': return makeToken(TokenKind::Comma, start);
        case ';': return makeToken(TokenKind::Semicolon, start);

        case '!':
            if (match('=')) return makeToken(TokenKind::BangEqual, start);
            return makeToken(TokenKind::Bang, start);

        case '=':
            if (match('=')) return makeToken(TokenKind::EqualEqual, start);
            return makeToken(TokenKind::Equal, start);

        case '>':
            if (match('=')) return makeToken(TokenKind::GreaterEqual, start);
            return makeToken(TokenKind::Greater, start);

        case '<':
            if (match('=')) return makeToken(TokenKind::LessEqual, start);
            return makeToken(TokenKind::Less, start);

        case '&':
            if (match('&')) return makeToken(TokenKind::AndAnd, start);
            break;

        case '|':
            if (match('|')) return makeToken(TokenKind::OrOr, start);
            break;
    }

    return makeToken(TokenKind::Unknown, start);
}

Token Lexer::nextToken() {
    skipWhitespaceAndComments();

    if (isAtEnd()) {
        return makeToken(TokenKind::Eof, position_);
    }

    char c = peek();

    // Number (possibly negative)
    if (c == '-' && std::isdigit(static_cast<unsigned char>(peekAhead(1)))) {
        return scanNumber();
    }

    if (std::isdigit(static_cast<unsigned char>(c))) {
        return scanNumber();
    }

    // Identifier or keyword
    if (isIdentifierStart(c)) {
        return scanIdentifierOrKeyword();
    }

    // Operators and punctuation
    return scanOperatorOrPunct();
}

} // namespace smollu
