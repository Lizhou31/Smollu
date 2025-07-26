/**
 * @file smollu_lexer.c
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Lexer for the Smollu language as defined in "Language Spec.md"
 * 
 * @version 0.1
 * @date 2025-07-26
 * 
 * @copyright Copyright (c) 2025 Lizhou
 * 
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smollu_compiler.h"

static inline int lex_peek(Lexer *l) {
    return l->pos < l->length ? l->src[l->pos] : EOF;
}

static inline int lex_peek_next(Lexer *l) {
    return (l->pos + 1) < l->length ? l->src[l->pos + 1] : EOF;
}

static inline int lex_advance(Lexer *l) {
    if (l->pos >= l->length) return EOF;
    int c = l->src[l->pos++];
    if (c == '\n') {
        l->line++;
        l->column = 1;
    } else {
        l->column++;
    }
    return c;
}

static void skip_whitespace_and_comments(Lexer *l) {
    for (;;) {
        int c = lex_peek(l);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            lex_advance(l);
            continue;
        }
        /* line comment starts with -- */
        if (c == '-' && lex_peek_next(l) == '-') {
            /* consume until newline or EOF */
            while (c != '\n' && c != EOF) {
                c = lex_advance(l);
            }
            continue;
        }
        break;
    }
}

static char *substr_dup(const char *src, size_t start, size_t end) {
    size_t len = end - start;
    char *out = (char *)malloc(len + 1);
    if (!out) {
        fprintf(stderr, "[Tokenizer] Out of memory while duplicating substring\n");
        exit(1);
    }
    memcpy(out, src + start, len);
    out[len] = '\0';
    return out;
}

static int is_identifier_start(int c) {
    return isalpha(c) || c == '_';
}
static int is_identifier_part(int c) {
    return isalnum(c) || c == '_';
}

static Token make_token(Lexer *l, TokenType type, size_t start, size_t end) {
    Token t;
    t.type   = type;
    t.lexeme = substr_dup(l->src, start, end);
    t.line   = l->line;
    t.column = l->column - (int)(end - start); /* approximate starting column */
    memset(&t.value, 0, sizeof(t.value));
    return t;
}

/* Keyword lookup */
static TokenType keyword_type(const char *lexeme) {
    if (strcmp(lexeme, "native") == 0)   return TOK_KW_NATIVE;
    if (strcmp(lexeme, "function") == 0) return TOK_KW_FUNCTION;
    if (strcmp(lexeme, "init") == 0)     return TOK_KW_INIT;
    if (strcmp(lexeme, "main") == 0)     return TOK_KW_MAIN;
    if (strcmp(lexeme, "local") == 0)    return TOK_KW_LOCAL;
    if (strcmp(lexeme, "while") == 0)    return TOK_KW_WHILE;
    if (strcmp(lexeme, "if") == 0)       return TOK_KW_IF;
    if (strcmp(lexeme, "else") == 0)     return TOK_KW_ELSE;
    return TOK_IDENTIFIER;
}

static Token scan_identifier_or_keyword(Lexer *l) {
    size_t start = l->pos;
    lex_advance(l); /* consume first char */
    while (is_identifier_part(lex_peek(l))) {
        lex_advance(l);
    }
    size_t end = l->pos;

    char *lexeme = substr_dup(l->src, start, end);

    /* Handle literals true/false/nil separately */
    if (strcmp(lexeme, "true") == 0 || strcmp(lexeme, "false") == 0) {
        Token tok = make_token(l, TOK_BOOL_LITERAL, start, end);
        tok.value.bool_val = (strcmp(lexeme, "true") == 0);
        free(lexeme); /* make_token duplicated already */
        return tok;
    }
    if (strcmp(lexeme, "nil") == 0) {
        Token tok = make_token(l, TOK_NIL_LITERAL, start, end);
        free(lexeme);
        return tok;
    }
    TokenType kw = keyword_type(lexeme);
    Token tok = make_token(l, kw, start, end);
    free(lexeme);
    return tok;
}

static Token scan_number(Lexer *l) {
    size_t start = l->pos;
    int has_minus = 0;
    if (lex_peek(l) == '-') {
        has_minus = 1;
        lex_advance(l);
    }
    while (isdigit(lex_peek(l))) {
        lex_advance(l);
    }
    int is_float = 0;
    if (lex_peek(l) == '.') {
        is_float = 1;
        lex_advance(l); /* consume '.' */
        while (isdigit(lex_peek(l))) {
            lex_advance(l);
        }
    }
    size_t end = l->pos;
    Token tok;
    if (is_float) {
        tok = make_token(l, TOK_FLOAT_LITERAL, start, end);
        tok.value.float_val = strtof(tok.lexeme, NULL);
    } else {
        tok = make_token(l, TOK_INT_LITERAL, start, end);
        tok.value.int_val = atoi(tok.lexeme);
    }
    (void)has_minus; /* sign captured in lexeme already */
    return tok;
}

static int match(Lexer *l, int expected) {
    if (lex_peek(l) != expected) return 0;
    lex_advance(l);
    return 1;
}

static Token scan_operator_or_punct(Lexer *l) {
    int c = lex_advance(l);
    size_t start = l->pos - 1;
    Token tok;
#define RET(tk_type) return make_token(l, tk_type, start, l->pos)
    switch (c) {
        case '+': RET(TOK_PLUS);
        case '-':
            /* Could be start of number, but we only get here if previous char wasn't '-' (comment) */
            RET(TOK_MINUS);
        case '*': RET(TOK_STAR);
        case '/': RET(TOK_SLASH);
        case '%': RET(TOK_PERCENT);

        case '(': RET(TOK_LPAREN);
        case ')': RET(TOK_RPAREN);
        case '{': RET(TOK_LBRACE);
        case '}': RET(TOK_RBRACE);
        case ',': RET(TOK_COMMA);
        case ';': RET(TOK_SEMICOLON);

        case '!':
            if (match(l, '=')) RET(TOK_BANG_EQUAL);
            else RET(TOK_BANG);
        case '=':
            if (match(l, '=')) RET(TOK_EQUAL_EQUAL);
            else RET(TOK_EQUAL);
        case '>':
            if (match(l, '=')) RET(TOK_GREATER_EQUAL);
            else RET(TOK_GREATER);
        case '<':
            if (match(l, '=')) RET(TOK_LESS_EQUAL);
            else RET(TOK_LESS);
        case '&':
            if (match(l, '&')) RET(TOK_AND_AND);
            break; /* unknown otherwise */
        case '|':
            if (match(l, '|')) RET(TOK_OR_OR);
            break;
    }
#undef RET
    tok = make_token(l, TOK_UNKNOWN, start, l->pos);
    return tok;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Public API implementation                                                  */
/* ──────────────────────────────────────────────────────────────────────────── */

void lexer_init(Lexer *lex, const char *source_code) {
    lex->src    = source_code;
    lex->pos    = 0;
    lex->length = strlen(source_code);
    lex->line   = 1;
    lex->column = 1;
}

void lexer_free(Lexer *lex) {
    (void)lex; /* nothing to free for now */
}

Token lexer_next(Lexer *lex) {
    skip_whitespace_and_comments(lex);
    if (lex->pos >= lex->length) {
        return make_token(lex, TOK_EOF, lex->pos, lex->pos);
    }

    int c = lex_peek(lex);

    /* Number (possibly signed) */
    if (c == '-' && isdigit(lex_peek_next(lex))) {
        return scan_number(lex);
    }
    if (isdigit(c)) {
        return scan_number(lex);
    }

    /* Identifier or keyword / literals */
    if (is_identifier_start(c)) {
        return scan_identifier_or_keyword(lex);
    }

    /* Operators / punctuation */
    return scan_operator_or_punct(lex);
}

const char *token_type_name(TokenType t) {
    switch (t) {
        case TOK_EOF:            return "EOF";
        case TOK_UNKNOWN:        return "UNKNOWN";
        case TOK_IDENTIFIER:     return "IDENTIFIER";
        case TOK_INT_LITERAL:    return "INT_LITERAL";
        case TOK_FLOAT_LITERAL:  return "FLOAT_LITERAL";
        case TOK_BOOL_LITERAL:   return "BOOL_LITERAL";
        case TOK_NIL_LITERAL:    return "NIL_LITERAL";

        case TOK_KW_NATIVE:     return "KW_NATIVE";
        case TOK_KW_FUNCTION:   return "KW_FUNCTION";
        case TOK_KW_INIT:       return "KW_INIT";
        case TOK_KW_MAIN:       return "KW_MAIN";
        case TOK_KW_LOCAL:      return "KW_LOCAL";
        case TOK_KW_WHILE:      return "KW_WHILE";
        case TOK_KW_IF:         return "KW_IF";
        case TOK_KW_ELSE:       return "KW_ELSE";

        case TOK_PLUS:          return "+";
        case TOK_MINUS:         return "-";
        case TOK_STAR:          return "*";
        case TOK_SLASH:         return "/";
        case TOK_PERCENT:       return "%";
        case TOK_EQUAL:         return "=";
        case TOK_EQUAL_EQUAL:   return "==";
        case TOK_BANG:          return "!";
        case TOK_BANG_EQUAL:    return "!=";
        case TOK_GREATER:       return ">";
        case TOK_GREATER_EQUAL: return ">=";
        case TOK_LESS:          return "<";
        case TOK_LESS_EQUAL:    return "<=";
        case TOK_AND_AND:       return "&&";
        case TOK_OR_OR:         return "||";
        case TOK_LPAREN:        return "(";
        case TOK_RPAREN:        return ")";
        case TOK_LBRACE:        return "{";
        case TOK_RBRACE:        return "}";
        case TOK_COMMA:         return ",";
        case TOK_SEMICOLON:     return ";";
        default:                return "<UNKNOWN TOKENTYPE>";
    }
}

void token_free(Token *tok) {
    if (tok && tok->lexeme) {
        free(tok->lexeme);
        tok->lexeme = NULL;
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Simple standalone test (compile with -DTEST_TOKENIZER)                     */
/* ──────────────────────────────────────────────────────────────────────────── */
#ifdef TEST_LEXER
int main(void) {
    const char *code =
        "-- Sample Smollu code\n"
        "init {\n"
            "native gpio_write(pin, value) -- native declaration\n"
            "LED_PIN = 1\n"
        "}\n"
        "function blink() {\n"
        "    local x = -42.5;\n"
        "    while (x < 0) { x = x + 1; }\n"
        "    if (x < 0) { x = x + 1; }\n"
        "    else if (x < 0) { x = x + 1; }\n"
        "    else { x = x + 1; }\n"
        "}\n"
        "main {\n"
        "    blink();\n"
        "}\n";

    Lexer lex;
    lexer_init(&lex, code);

    for (;;) {
        Token tok = lexer_next(&lex);
        printf("[%3d:%3d] %-15s '%s'\n", tok.line, tok.column, token_type_name(tok.type), tok.lexeme);
        if (tok.type == TOK_EOF) {
            token_free(&tok);
            break;
        }
        token_free(&tok);
    }
    return 0;
}
#endif
