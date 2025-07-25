#include <criterion/criterion.h>
#include <string.h>

#include "../src/components/compiler/smollu_compiler.h"

/* Helper wrappers to make assertions concise */
static void expect_type(Token tok, TokenType expected) {
    cr_assert_eq(tok.type, expected,
                 "Expected token type %s, got %s",
                 token_type_name(expected), token_type_name(tok.type));
}
static void expect_lexeme(Token tok, const char *expected) {
    cr_assert_str_eq(tok.lexeme, expected,
                     "Expected lexeme '%s', got '%s'", expected, tok.lexeme);
}

Test(lexer, basic_syntax) {
    const char *source = "init { native foo(a, b) }";
    Lexer lex; lexer_init(&lex, source);
    Token tok;

    tok = lexer_next(&lex); expect_type(tok, TOK_KW_INIT);  expect_lexeme(tok, "init"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_LBRACE);                               token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_NATIVE);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "foo"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_LPAREN);                               token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "a");  token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_COMMA);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "b");  token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_RPAREN);                               token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_RBRACE);                               token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EOF);                                  token_free(&tok);

    lexer_free(&lex);
}

Test(lexer, literals) {
    const char *source =
        "local x = -42; "
        "local y = 3.14; "
        "z = 1.;"
        "local t = true; "
        "local f = false; "
        "local n = nil;";

    Lexer lex; lexer_init(&lex, source);
    Token tok;

    /* local x = -42; */
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_LOCAL);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "x"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_INT_LITERAL);  cr_assert_eq(tok.value.int_val, -42); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    /* local y = 3.14; */
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_LOCAL);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "y"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_FLOAT_LITERAL);
    cr_assert_float_eq(tok.value.float_val, 3.14f, 1e-3);
    token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    /* z = 1.; */
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "z"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_FLOAT_LITERAL);
    cr_assert_float_eq(tok.value.float_val, 1.0f, 1e-3);
    token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    /* local t = true; */
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_LOCAL);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "t"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_BOOL_LITERAL); cr_assert_eq(tok.value.bool_val, 1); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    /* local f = false; */
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_LOCAL);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "f"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_BOOL_LITERAL); cr_assert_eq(tok.value.bool_val, 0); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    /* local n = nil; */
    tok = lexer_next(&lex); expect_type(tok, TOK_KW_LOCAL);                            token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_IDENTIFIER); expect_lexeme(tok, "n"); token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_EQUAL);                                token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_NIL_LITERAL);                          token_free(&tok);
    tok = lexer_next(&lex); expect_type(tok, TOK_SEMICOLON);                            token_free(&tok);

    tok = lexer_next(&lex); expect_type(tok, TOK_EOF);                                  token_free(&tok);

    lexer_free(&lex);
} 