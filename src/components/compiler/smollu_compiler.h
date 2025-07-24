#ifndef SMOLLU_COMPILER_H
#define SMOLLU_COMPILER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> /* for size_t */

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Token types                                                                */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef enum {
    /* Special */
    TOK_EOF = 0,
    TOK_UNKNOWN,

    /* Literals */
    TOK_IDENTIFIER,
    TOK_INT_LITERAL,
    TOK_FLOAT_LITERAL,
    TOK_BOOL_LITERAL, /* true / false */
    TOK_NIL_LITERAL,  /* nil */

    /* Keywords */
    TOK_KW_NATIVE,
    TOK_KW_FUNCTION,
    TOK_KW_INIT,
    TOK_KW_MAIN,
    TOK_KW_LOCAL,
    TOK_KW_WHILE,
    TOK_KW_IF,
    TOK_KW_ELSE,

    /* Operators */
    TOK_PLUS,          /* + */
    TOK_MINUS,         /* - */
    TOK_STAR,          /* * */
    TOK_SLASH,         /* / */
    TOK_PERCENT,       /* % */

    TOK_EQUAL,         /* = */
    TOK_EQUAL_EQUAL,   /* == */
    TOK_BANG,          /* ! */
    TOK_BANG_EQUAL,    /* != */

    TOK_GREATER,       /* > */
    TOK_GREATER_EQUAL, /* >= */
    TOK_LESS,          /* < */
    TOK_LESS_EQUAL,    /* <= */

    TOK_AND_AND,       /* && */
    TOK_OR_OR,         /* || */

    /* Delimiters */
    TOK_LPAREN,        /* ( */
    TOK_RPAREN,        /* ) */
    TOK_LBRACE,        /* { */
    TOK_RBRACE,        /* } */
    TOK_COMMA,         /* , */
    TOK_SEMICOLON      /* ; */
} TokenType;

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Token structure                                                            */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct Token {
    TokenType type;   /* kind of token */
    char     *lexeme; /* malloc'd, null-terminated slice of the original source */
    int       line;   /* 1-based line number */
    int       column; /* 1-based column number (start position) */
    union {
        int   int_val;
        float float_val;
        int   bool_val; /* 0 = false, 1 = true */
    } value;          /* literal value for numeric / boolean tokens */
} Token;

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Lexer structure                                                            */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct Lexer {
    const char *src;   /* original source buffer (not owned) */
    size_t      pos;   /* byte offset into src */
    size_t      length;/* cached strlen(src) */
    int         line;  /* current line (1-based) */
    int         column;/* current column (1-based) */
} Lexer;

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Public API                                                                 */
/* ──────────────────────────────────────────────────────────────────────────── */

/* Initialise a lexer to scan the given null-terminated source code string. */
void  lexer_init(Lexer *lex, const char *source_code);

/* Release any internal resources (currently a no-op, but call for symmetry). */
void  lexer_free(Lexer *lex);

/* Scan and return the next token from the input stream. */
Token lexer_next(Lexer *lex);

/* Return a human-readable string for a given TokenType (for debugging). */
const char *token_type_name(TokenType t);

/* Free memory owned by a token (its lexeme). */
void  token_free(Token *tok);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SMOLLU_COMPILER_H */