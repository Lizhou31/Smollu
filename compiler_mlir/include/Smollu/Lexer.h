/**
 * @file Lexer.h
 * @author Lizhou (lisie31s@gmail.com)
 * @brief C++ Lexer for Smollu with MLIR location tracking support
 *
 * @version 0.2
 * @date 2025-10-03
 *
 * @copyright Copyright (c) 2025 Lizhou
 *
 */

#ifndef SMOLLU_LEXER_H
#define SMOLLU_LEXER_H

#include <memory>
#include <string>
#include <string_view>
#include <variant>

namespace smollu {

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Token Types                                                                */
/* ──────────────────────────────────────────────────────────────────────────── */

enum class TokenKind {
    // Special
    Eof,
    Unknown,

    // Literals
    Identifier,
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    NilLiteral,

    // Keywords
    KwNative,
    KwFunction,
    KwFunctions,
    KwReturn,
    KwInit,
    KwMain,
    KwLocal,
    KwWhile,
    KwIf,
    KwElif,
    KwElse,

    // Operators
    Plus,          // +
    Minus,         // -
    Star,          // *
    Slash,         // /
    Percent,       // %

    Equal,         // =
    EqualEqual,    // ==
    Bang,          // !
    BangEqual,     // !=

    Greater,       // >
    GreaterEqual,  // >=
    Less,          // <
    LessEqual,     // <=

    AndAnd,        // &&
    OrOr,          // ||

    // Delimiters
    LParen,        // (
    RParen,        // )
    LBrace,        // {
    RBrace,        // }
    Comma,         // ,
    Semicolon      // ;
};

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Source Location                                                            */
/* ──────────────────────────────────────────────────────────────────────────── */

/// Represents a location in source code (file, line, column)
struct SourceLocation {
    std::string filename;
    unsigned line;        // 1-based
    unsigned column;      // 1-based
    size_t offset;        // 0-based byte offset in source

    SourceLocation() : line(1), column(1), offset(0) {}

    SourceLocation(std::string_view file, unsigned ln, unsigned col, size_t off)
        : filename(file), line(ln), column(col), offset(off) {}

    /// Format as "filename:line:column"
    std::string toString() const;
};

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Token                                                                      */
/* ──────────────────────────────────────────────────────────────────────────── */

/// Represents a lexical token with location information
class Token {
public:
    using ValueType = std::variant<std::monostate, int, float, bool>;

    Token() : kind_(TokenKind::Eof), location_() {}

    Token(TokenKind kind, SourceLocation loc, std::string_view lexeme,
          ValueType value = std::monostate{})
        : kind_(kind), location_(std::move(loc)), lexeme_(lexeme), value_(value) {}

    // Accessors
    TokenKind getKind() const { return kind_; }
    const SourceLocation& getLocation() const { return location_; }
    std::string_view getLexeme() const { return lexeme_; }
    const ValueType& getValue() const { return value_; }

    // Type-specific value accessors
    int getIntValue() const { return std::get<int>(value_); }
    float getFloatValue() const { return std::get<float>(value_); }
    bool getBoolValue() const { return std::get<bool>(value_); }

    // Utility
    bool is(TokenKind kind) const { return kind_ == kind; }
    bool isNot(TokenKind kind) const { return kind_ != kind; }
    bool isOneOf(TokenKind k1, TokenKind k2) const { return is(k1) || is(k2); }
    template<typename... Ts>
    bool isOneOf(TokenKind k1, TokenKind k2, Ts... ks) const {
        return is(k1) || isOneOf(k2, ks...);
    }

    /// Get human-readable token kind name
    std::string_view getKindName() const;

private:
    TokenKind kind_;
    SourceLocation location_;
    std::string_view lexeme_;  // Points into source buffer
    ValueType value_;
};

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Lexer                                                                      */
/* ──────────────────────────────────────────────────────────────────────────── */

/// Lexer for the Smollu language
class Lexer {
public:
    /// Construct a lexer from source code and filename
    Lexer(std::string source, std::string filename);

    /// Get the next token from the input
    Token nextToken();

    /// Peek at the current position without advancing
    char peek() const;

    /// Peek ahead n characters
    char peekAhead(size_t n = 1) const;

    /// Get current location
    SourceLocation getCurrentLocation() const;

    /// Get the source buffer (for string_view references)
    const std::string& getSource() const { return source_; }

private:
    // Character operations
    char advance();
    bool match(char expected);
    bool isAtEnd() const;

    // Whitespace and comments
    void skipWhitespaceAndComments();

    // Token scanning
    Token scanIdentifierOrKeyword();
    Token scanNumber();
    Token scanOperatorOrPunct();

    // Helpers
    Token makeToken(TokenKind kind, size_t start);
    Token makeToken(TokenKind kind, size_t start, Token::ValueType value);
    SourceLocation makeLocation(size_t offset) const;

    static bool isIdentifierStart(char c);
    static bool isIdentifierPart(char c);
    static TokenKind keywordKind(std::string_view lexeme);

    std::string source_;     // Owned source code buffer
    std::string filename_;   // Source filename
    size_t position_;        // Current byte offset
    unsigned line_;          // Current line (1-based)
    unsigned column_;        // Current column (1-based)
};

} // namespace smollu

#endif // SMOLLU_LEXER_H
