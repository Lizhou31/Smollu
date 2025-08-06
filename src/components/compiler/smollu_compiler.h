/**
 * @file smollu_compiler.h
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Public API for the Smollu compiler
 * 
 * @version 0.1
 * @date 2025-07-26
 * 
 * @copyright Copyright (c) 2025 Lizhou
 * 
 */
#ifndef SMOLLU_COMPILER_H
#define SMOLLU_COMPILER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> /* for size_t */
#include <stdint.h> /* for uint8_t */
#include <stdio.h>  /* for FILE */

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
    TOK_KW_FUNCTIONS,
    TOK_KW_RETURN,
    TOK_KW_INIT,
    TOK_KW_MAIN,
    TOK_KW_LOCAL,
    TOK_KW_WHILE,
    TOK_KW_IF,
    TOK_KW_ELIF,
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

/* ──────────────────────────────────────────────────────────────────────────── */
/*  AST definitions                                                            */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef enum {
    AST_PROGRAM,
    AST_BLOCK,
    AST_INT_LITERAL,
    AST_FLOAT_LITERAL,
    AST_BOOL_LITERAL,
    AST_NIL_LITERAL,
    AST_IDENTIFIER,
    AST_BINARY,
    AST_UNARY,
    AST_ASSIGNMENT,
    AST_WHILE,
    AST_IF,
    AST_FUNCTION_CALL,
    AST_NATIVE_CALL,
    AST_FUNCTION_DEF,
    AST_RETURN,
    AST_PARAMETER_LIST
} NodeType;

typedef struct ASTNode ASTNode;
struct ASTNode {
    NodeType type;
    int      line;
    int      column;
    ASTNode *next; /* linked-list for sequential nodes (e.g., statements) */
    union {
        /* Literal values */
        int   int_val;
        float float_val;
        int   bool_val;
        char *identifier; /* malloc'd */

        struct {               /* Unary expression */
            TokenType op;      /* operator token type */
            ASTNode  *expr;    /* operand */
        } unary;
        struct {               /* Binary expression */
            TokenType op;      /* operator token type */
            ASTNode  *left;
            ASTNode  *right;
        } binary;
        struct {               /* Assignment */
            char    *name;     /* identifier */
            int      is_local; /* boolean */
            ASTNode *value;    /* expression */
        } assign;
        struct {               /* While statement */
            ASTNode *condition;
            ASTNode *body;     /* AST_BLOCK */
        } while_stmt;
        struct {
            ASTNode *init;      /* AST_BLOCK */
            ASTNode *main;      /* AST_BLOCK */
            ASTNode *functions; /* AST_BLOCK containing function definitions */
        } program;
        struct {               /* Return statement */
            ASTNode *value;    /* expression */
        } return_stmt;
        struct {                /* If statement */
            ASTNode *condition; /* AST_BINARY */
            ASTNode *then_body; /* AST_BLOCK */
            ASTNode *else_body; /* NULL or AST_BLOCK or AST_IF */
        } if_stmt;
        struct {               /* Block */
            ASTNode *stmts;    /* linked-list of statements */
        } block;
        struct {               /* Function call */
            char    *name;     /* function name */
            ASTNode *args;     /* linked-list of argument expressions */
        } func_call;
        struct {               /* Native call */
            char    *name;     /* native function name */
            ASTNode *args;     /* linked-list of argument expressions */
        } native_call;
        struct {               /* Function definition */
            char    *name;     /* function name */
            ASTNode *params;   /* linked-list of parameter names (identifiers) */
            ASTNode *body;     /* AST_BLOCK */
        } func_def;
        struct {               /* Parameter list */
            char    *param_name; /* parameter name */
        } param;
    } as;
};

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Parser interface                                                           */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct Parser {
    Lexer *lex;      /* external lexer provided by the caller */
    Token  current;  /* current lookahead token */
} Parser;

/* Initialize parser with an already configured lexer */
void     parser_init(Parser *p, Lexer *lex);

/* Free resources held by parser (including current token) */
void     parser_free(Parser *p);

/* Parse an entire program and return its AST root (AST_PROGRAM). */
ASTNode *parse_program(Parser *p);

/* Recursively free an AST tree. */
void     ast_free(ASTNode *node);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Bytecode code generator interface                                        */
/* ──────────────────────────────────────────────────────────────────────────── */

/* Generate bytecode from an AST root node. */
int smollu_generate_bytecode(ASTNode *root, uint8_t device_id, uint8_t version, uint8_t **out_buf, size_t *out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Public Compiler API                                                        */
/* ──────────────────────────────────────────────────────────────────────────── */
int smollu_compile(FILE *in, FILE *out, FILE *ast_out);

#endif /* SMOLLU_COMPILER_H */