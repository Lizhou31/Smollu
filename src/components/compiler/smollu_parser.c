/**
 * @file smollu_parser.c
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Recursive-descent parser building an AST for Smollu.
 * Depends on the lexer defined in smollu_lexer.c and public API in smollu_compiler.h
 * 
 * @version 0.1
 * @date 2025-07-26
 * 
 * @copyright Copyright (c) 2025 Lizhou
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "smollu_compiler.h"

//#define TEST_PARSER

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Internal helpers                                                           */
/* ──────────────────────────────────────────────────────────────────────────── */

static ASTNode *new_node(NodeType type, int line, int column) {
    ASTNode *n = (ASTNode *)calloc(1, sizeof(ASTNode));
    if (!n) {
        fprintf(stderr, "[Parser] Out of memory allocating AST node\n");
        exit(1);
    }
    n->type   = type;
    n->line   = line;
    n->column = column;
    return n;
}

static void parser_advance(Parser *p) {
    token_free(&p->current);
    p->current = lexer_next(p->lex);
}

static int parser_check(Parser *p, TokenType t) {
    return p->current.type == t;
}

static int parser_match(Parser *p, TokenType t) {
    if (parser_check(p, t)) {
        parser_advance(p);
        return 1;
    }
    return 0;
}

static void parser_expect(Parser *p, TokenType t, const char *msg) {
    if (!parser_match(p, t)) {
        fprintf(stderr, "[Parser] Error at %d:%d: expected %s (%s)\n", p->current.line, p->current.column, msg, token_type_name(t));
        exit(1);
    }
}

/* Forward declarations for expression parsing (precedence climbing) */
static ASTNode *parse_expression(Parser *p);
static ASTNode *parse_argument_list(Parser *p);

/* ─────────────────── Function call helpers --------------------------------- */
static ASTNode *parse_argument_list(Parser *p) {
    if (parser_check(p, TOK_RPAREN)) {
        return NULL; /* empty argument list */
    }
    
    ASTNode *args = NULL;
    ASTNode *last = NULL;
    
    do {
        ASTNode *arg = parse_expression(p);
        if (!args) {
            args = arg;
            last = arg;
        } else {
            last->next = arg;
            last = arg;
        }
    } while (parser_match(p, TOK_COMMA));
    
    return args;
}

/* ─────────────────── Primary ------------------------------------------------ */
static ASTNode *parse_primary(Parser *p) {
    Token tok = p->current;
    switch (tok.type) {
        case TOK_INT_LITERAL: {
            ASTNode *n = new_node(AST_INT_LITERAL, tok.line, tok.column);
            n->as.int_val = tok.value.int_val;
            parser_advance(p);
            return n;
        }
        case TOK_FLOAT_LITERAL: {
            ASTNode *n = new_node(AST_FLOAT_LITERAL, tok.line, tok.column);
            n->as.float_val = tok.value.float_val;
            parser_advance(p);
            return n;
        }
        case TOK_BOOL_LITERAL: {
            ASTNode *n = new_node(AST_BOOL_LITERAL, tok.line, tok.column);
            n->as.bool_val = tok.value.bool_val;
            parser_advance(p);
            return n;
        }
        case TOK_NIL_LITERAL: {
            ASTNode *n = new_node(AST_NIL_LITERAL, tok.line, tok.column);
            parser_advance(p);
            return n;
        }
        case TOK_IDENTIFIER: {
            ASTNode *n = new_node(AST_IDENTIFIER, tok.line, tok.column);
            n->as.identifier = strdup(tok.lexeme);
            parser_advance(p);
            return n;
        }
        case TOK_LPAREN: {
            parser_advance(p);
            ASTNode *expr = parse_expression(p);
            parser_expect(p, TOK_RPAREN, ")");
            return expr;
        }
        default:
            fprintf(stderr, "[Parser] Unexpected token %s at %d:%d\n", token_type_name(tok.type), tok.line, tok.column);
            exit(1);
    }
}

/* ─────────────────── Postfix (function calls) ------------------------------ */
static ASTNode *parse_postfix(Parser *p) {
    ASTNode *expr = parse_primary(p);
    
    while (parser_check(p, TOK_LPAREN)) {
        if (expr->type != AST_IDENTIFIER) {
            fprintf(stderr, "[Parser] Function call on non-identifier at %d:%d\n", 
                    p->current.line, p->current.column);
            exit(1);
        }
        
        Token tok = p->current;
        parser_advance(p); /* consume '(' */
        ASTNode *args = parse_argument_list(p);
        parser_expect(p, TOK_RPAREN, ")");
        
        ASTNode *call = new_node(AST_FUNCTION_CALL, tok.line, tok.column);
        call->as.func_call.name = strdup(expr->as.identifier);
        call->as.func_call.args = args;
        
        ast_free(expr); /* free the original identifier node */
        expr = call;
    }
    
    return expr;
}

/* ─────────────────── Unary -------------------------------------------------- */
static ASTNode *parse_unary(Parser *p) {
    if (parser_check(p, TOK_BANG) || parser_check(p, TOK_MINUS)) {
        TokenType op = p->current.type;
        Token tok = p->current;
        parser_advance(p);
        ASTNode *operand = parse_unary(p);
        ASTNode *n = new_node(AST_UNARY, tok.line, tok.column);
        n->as.unary.op = op;
        n->as.unary.expr = operand;
        return n;
    }
    return parse_postfix(p);
}

/* Helper to create binary expression nodes */
static ASTNode *make_binary(Parser *p, ASTNode *left, TokenType op, ASTNode *right) {
    ASTNode *n = new_node(AST_BINARY, left->line, left->column);
    n->as.binary.op = op;
    n->as.binary.left = left;
    n->as.binary.right = right;
    return n;
}

/* ─────────────────── Binary precedence levels ------------------------------ */
static ASTNode *parse_factor(Parser *p); // * / %
static ASTNode *parse_term(Parser *p);   // + -
static ASTNode *parse_comparison(Parser *p); // < > <= >=
static ASTNode *parse_equality(Parser *p);  // == !=
static ASTNode *parse_logic_and(Parser *p); // &&
static ASTNode *parse_logic_or(Parser *p);  // ||

static ASTNode *parse_factor(Parser *p) {
    ASTNode *left = parse_unary(p);
    while (parser_check(p, TOK_STAR) || parser_check(p, TOK_SLASH) || parser_check(p, TOK_PERCENT)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_unary(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_term(Parser *p) {
    ASTNode *left = parse_factor(p);
    while (parser_check(p, TOK_PLUS) || parser_check(p, TOK_MINUS)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_factor(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_comparison(Parser *p) {
    ASTNode *left = parse_term(p);
    while (parser_check(p, TOK_LESS) || parser_check(p, TOK_LESS_EQUAL) ||
           parser_check(p, TOK_GREATER) || parser_check(p, TOK_GREATER_EQUAL)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_term(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_equality(Parser *p) {
    ASTNode *left = parse_comparison(p);
    while (parser_check(p, TOK_EQUAL_EQUAL) || parser_check(p, TOK_BANG_EQUAL)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_comparison(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_logic_and(Parser *p) {
    ASTNode *left = parse_equality(p);
    while (parser_check(p, TOK_AND_AND)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_equality(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_logic_or(Parser *p) {
    ASTNode *left = parse_logic_and(p);
    while (parser_check(p, TOK_OR_OR)) {
        TokenType op = p->current.type;
        parser_advance(p);
        ASTNode *right = parse_logic_and(p);
        left = make_binary(p, left, op, right);
    }
    return left;
}
static ASTNode *parse_expression(Parser *p) {
    return parse_logic_or(p);
}

/* ─────────────────── Statements -------------------------------------------- */
static ASTNode *parse_block(Parser *p);

static ASTNode *parse_while(Parser *p) {
    Token tok = p->current; /* TOK_WHILE */
    parser_advance(p);
    parser_expect(p, TOK_LPAREN, "(");
    ASTNode *cond = parse_expression(p);
    parser_expect(p, TOK_RPAREN, ")");
    ASTNode *body = parse_block(p);

    ASTNode *n = new_node(AST_WHILE, tok.line, tok.column);
    n->as.while_stmt.condition = cond;
    n->as.while_stmt.body      = body;
    return n;
}

static ASTNode *parse_if(Parser *p) {
    Token tok = p->current; /* TOK_IF */
    parser_advance(p);
    parser_expect(p, TOK_LPAREN, "(");
    ASTNode *cond = parse_expression(p);
    parser_expect(p, TOK_RPAREN, ")");
    ASTNode *then_body = parse_block(p);
    ASTNode *else_body = NULL;
    if(parser_check(p, TOK_KW_ELIF)) {
        else_body = parse_if(p);
    } else if (parser_check(p, TOK_KW_ELSE)) {
        parser_advance(p);
        else_body = parse_block(p);
    }

    ASTNode *n = new_node(AST_IF, tok.line, tok.column);
    n->as.if_stmt.condition = cond;
    n->as.if_stmt.then_body = then_body;
    n->as.if_stmt.else_body = else_body;
    return n;
}

static ASTNode *parse_assignment(Parser *p, int is_local) {
    Token ident_tok = p->current; /* identifier */
    char *name = strdup(ident_tok.lexeme);
    parser_expect(p, TOK_IDENTIFIER, "identifier");
    parser_expect(p, TOK_EQUAL, "=");
    ASTNode *value = parse_expression(p);
    parser_expect(p, TOK_SEMICOLON, ";");

    ASTNode *n = new_node(AST_ASSIGNMENT, ident_tok.line, ident_tok.column);
    n->as.assign.name = name;
    n->as.assign.is_local = is_local;
    n->as.assign.value = value;
    return n;
}

static ASTNode *parse_statement(Parser *p) {
    if (parser_check(p, TOK_KW_WHILE)) {
        return parse_while(p);
    }
    if (parser_check(p, TOK_KW_IF)) {
        return parse_if(p);
    }
    if (parser_check(p, TOK_KW_NATIVE)) {
        Token tok = p->current;
        parser_advance(p); /* consume 'native' */
        
        if (!parser_check(p, TOK_IDENTIFIER)) {
            fprintf(stderr, "[Parser] Expected identifier after 'native' at %d:%d\n", 
                    p->current.line, p->current.column);
            exit(1);
        }
        
        char *name = strdup(p->current.lexeme);
        parser_advance(p); /* consume identifier */
        parser_expect(p, TOK_LPAREN, "(");
        ASTNode *args = parse_argument_list(p);
        parser_expect(p, TOK_RPAREN, ")");
        parser_expect(p, TOK_SEMICOLON, ";");
        
        ASTNode *native_call = new_node(AST_NATIVE_CALL, tok.line, tok.column);
        native_call->as.native_call.name = name;
        native_call->as.native_call.args = args;
        return native_call;
    }
    if (parser_check(p, TOK_KW_LOCAL)) {
        parser_advance(p);
        return parse_assignment(p, 1);
    }
    if (parser_check(p, TOK_IDENTIFIER)) {
        /* Lookahead for assignment or function call */
        size_t saved_pos   = p->lex->pos;
        int    saved_line  = p->lex->line;
        int    saved_col   = p->lex->column;

        Token next_tok = lexer_next(p->lex);

        /* restore lexer state */
        p->lex->pos    = saved_pos;
        p->lex->line   = saved_line;
        p->lex->column = saved_col;

        if (next_tok.type == TOK_EQUAL) {
            token_free(&next_tok);
            return parse_assignment(p, 0);
        } else if (next_tok.type == TOK_LPAREN) {
            /* Function call statement */
            Token ident_tok = p->current;
            char *name = strdup(ident_tok.lexeme);
            parser_advance(p); /* consume identifier */
            parser_advance(p); /* consume '(' */
            ASTNode *args = parse_argument_list(p);
            parser_expect(p, TOK_RPAREN, ")");
            parser_expect(p, TOK_SEMICOLON, ";");
            
            ASTNode *func_call = new_node(AST_FUNCTION_CALL, ident_tok.line, ident_tok.column);
            func_call->as.func_call.name = name;
            func_call->as.func_call.args = args;
            token_free(&next_tok);
            return func_call;
        }
        token_free(&next_tok);
        /* else fallthrough to expression statement */
    }

    /* Expression statement */
    ASTNode *expr = parse_expression(p);
    parser_expect(p, TOK_SEMICOLON, ";");
    return expr; /* treat bare expression as node */
}

static ASTNode *append_statement(ASTNode *list, ASTNode *stmt) {
    if (!list) return stmt;
    ASTNode *iter = list;
    while (iter->next) iter = iter->next;
    iter->next = stmt;
    return list;
}

static ASTNode *parse_block(Parser *p) {
    parser_expect(p, TOK_LBRACE, "{");
    ASTNode *stmts = NULL;
    while (!parser_check(p, TOK_RBRACE) && !parser_check(p, TOK_EOF)) {
        ASTNode *s = parse_statement(p);
        stmts = append_statement(stmts, s);
    }
    parser_expect(p, TOK_RBRACE, "}");
    ASTNode *block = new_node(AST_BLOCK, p->current.line, p->current.column);
    block->as.block.stmts = stmts;
    return block;
}

/* ─────────────────── Program ------------------------------------------------ */
static ASTNode *parse_init(Parser *p) {
    parser_expect(p, TOK_LBRACE, "{");
    ASTNode *stmts = NULL;
    while (!parser_check(p, TOK_RBRACE)) {
        ASTNode *s = parse_statement(p);
        stmts = append_statement(stmts, s);
    }
    parser_expect(p, TOK_RBRACE, "}");
    ASTNode *init = new_node(AST_BLOCK, 1, 1);
    init->as.block.stmts = stmts;
    return init;
}

static ASTNode *parse_main(Parser *p) {
    parser_expect(p, TOK_LBRACE, "{");
    ASTNode *stmts = NULL;
    while (!parser_check(p, TOK_RBRACE)) {
        ASTNode *s = parse_statement(p);
        stmts = append_statement(stmts, s);
    }
    parser_expect(p, TOK_RBRACE, "}");
    ASTNode *main = new_node(AST_BLOCK, 1, 1);
    main->as.block.stmts = stmts;
    return main;
}

static ASTNode *parse_parameter_list(Parser *p) {
    if (parser_check(p, TOK_RPAREN)) {
        return NULL; /* empty parameter list */
    }
    
    ASTNode *params = NULL;
    ASTNode *last = NULL;
    
    do {
        if (!parser_check(p, TOK_IDENTIFIER)) {
            fprintf(stderr, "[Parser] Expected parameter name at %d:%d\n", 
                    p->current.line, p->current.column);
            exit(1);
        }
        
        ASTNode *param = new_node(AST_PARAMETER_LIST, p->current.line, p->current.column);
        param->as.param.param_name = strdup(p->current.lexeme);
        parser_advance(p);
        
        if (!params) {
            params = param;
            last = param;
        } else {
            last->next = param;
            last = param;
        }
    } while (parser_match(p, TOK_COMMA));
    
    return params;
}

static ASTNode *parse_function_def(Parser *p) {
    Token tok = p->current; /* TOK_KW_FUNCTION */
    parser_advance(p);
    
    if (!parser_check(p, TOK_IDENTIFIER)) {
        fprintf(stderr, "[Parser] Expected function name at %d:%d\n", 
                p->current.line, p->current.column);
        exit(1);
    }
    
    char *name = strdup(p->current.lexeme);
    parser_advance(p);
    
    parser_expect(p, TOK_LPAREN, "(");
    ASTNode *params = parse_parameter_list(p);
    parser_expect(p, TOK_RPAREN, ")");
    ASTNode *body = parse_block(p);
    
    ASTNode *func_def = new_node(AST_FUNCTION_DEF, tok.line, tok.column);
    func_def->as.func_def.name = name;
    func_def->as.func_def.params = params;
    func_def->as.func_def.body = body;
    return func_def;
}

static ASTNode *parse_functions(Parser *p) {
    parser_expect(p, TOK_LBRACE, "{");
    ASTNode *funcs = NULL;
    while (!parser_check(p, TOK_RBRACE) && !parser_check(p, TOK_EOF)) {
        if (parser_check(p, TOK_KW_FUNCTION)) {
            ASTNode *func_def = parse_function_def(p);
            funcs = append_statement(funcs, func_def);
        } else {
            fprintf(stderr, "[Parser] Expected 'function' in functions section at %d:%d\n", 
                    p->current.line, p->current.column);
            exit(1);
        }
    }
    parser_expect(p, TOK_RBRACE, "}");
    ASTNode *functions_block = new_node(AST_BLOCK, 1, 1);
    functions_block->as.block.stmts = funcs;
    return functions_block;
}

ASTNode *parse_program(Parser *p) {
    parser_expect(p, TOK_KW_INIT, "init");
    ASTNode *init = parse_init(p);

    parser_expect(p, TOK_KW_MAIN, "main");
    ASTNode *main = parse_main(p);

    parser_expect(p, TOK_KW_FUNCTIONS, "functions");
    ASTNode *functions = parse_functions(p);

    ASTNode *root = new_node(AST_PROGRAM, 1, 1);
    root->as.program.init = init;
    root->as.program.main = main;
    root->as.program.functions = functions;
    return root;
}

/* ─────────────────── Public interface -------------------------------------- */
void parser_init(Parser *p, Lexer *lex) {
    p->lex = lex;
    p->current = lexer_next(lex);
}

void parser_free(Parser *p) {
    token_free(&p->current);
}

/* Recursively free AST */
void ast_free(ASTNode *node) {
    if (!node) return;
    switch (node->type) {
        case AST_INT_LITERAL:
        case AST_FLOAT_LITERAL:
        case AST_BOOL_LITERAL:
        case AST_NIL_LITERAL:
            break;
        case AST_IDENTIFIER:
            free(node->as.identifier);
            break;
        case AST_UNARY:
            ast_free(node->as.unary.expr);
            break;
        case AST_BINARY:
            ast_free(node->as.binary.left);
            ast_free(node->as.binary.right);
            break;
        case AST_ASSIGNMENT:
            free(node->as.assign.name);
            ast_free(node->as.assign.value);
            break;
        case AST_WHILE:
            ast_free(node->as.while_stmt.condition);
            ast_free(node->as.while_stmt.body);
            break;
        case AST_IF:
            ast_free(node->as.if_stmt.condition);
            ast_free(node->as.if_stmt.then_body);
            ast_free(node->as.if_stmt.else_body);
            break;
        case AST_BLOCK:
            while (node->as.block.stmts) {
                ASTNode *next = node->as.block.stmts->next;
                ast_free(node->as.block.stmts);
                node->as.block.stmts = next;
            }
            break;
        case AST_PROGRAM: {
            ast_free(node->as.program.init);
            ast_free(node->as.program.main);
            ast_free(node->as.program.functions);
            break;
        }
        case AST_FUNCTION_CALL: {
            free(node->as.func_call.name);
            ASTNode *arg = node->as.func_call.args;
            while (arg) {
                ASTNode *next = arg->next;
                ast_free(arg);
                arg = next;
            }
            break;
        }
        case AST_NATIVE_CALL: {
            free(node->as.native_call.name);
            ASTNode *arg = node->as.native_call.args;
            while (arg) {
                ASTNode *next = arg->next;
                ast_free(arg);
                arg = next;
            }
            break;
        }
        case AST_FUNCTION_DEF: {
            free(node->as.func_def.name);
            ASTNode *param = node->as.func_def.params;
            while (param) {
                ASTNode *next = param->next;
                ast_free(param);
                param = next;
            }
            ast_free(node->as.func_def.body);
            break;
        }
        case AST_PARAMETER_LIST: {
            free(node->as.param.param_name);
            break;
        }
        default:
            break;
    }
    free(node);
}

/* ─────────────────── Test harness ------------------------------------------ */
#ifdef TEST_PARSER
static void print_indent(int n) { while (n--) printf("  "); }

static void print_ast(ASTNode *n, int depth) {
    if (!n) return;
    print_indent(depth);
    switch (n->type) {
        case AST_PROGRAM:
            printf("Program\n");
            break;
        case AST_BLOCK:
            printf("Block\n");
            break;
        case AST_INT_LITERAL:
            printf("Int %d\n", n->as.int_val);
            break;
        case AST_FLOAT_LITERAL:
            printf("Float %f\n", n->as.float_val);
            break;
        case AST_BOOL_LITERAL:
            printf("Bool %d\n", n->as.bool_val);
            break;
        case AST_NIL_LITERAL:
            printf("Nil\n");
            break;
        case AST_IDENTIFIER:
            printf("Identifier %s\n", n->as.identifier);
            break;
        case AST_UNARY:
            printf("Unary %s\n", token_type_name(n->as.unary.op));
            break;
        case AST_BINARY:
            printf("Binary %s\n", token_type_name(n->as.binary.op));
            break;
        case AST_ASSIGNMENT:
            printf("Assign %s (local=%d)\n", n->as.assign.name, n->as.assign.is_local);
            break;
        case AST_WHILE:
            printf("While\n");
            break;
        case AST_IF:
            printf("If\n");
            break;
        case AST_FUNCTION_CALL:
            printf("FunctionCall %s\n", n->as.func_call.name);
            break;
        case AST_NATIVE_CALL:
            printf("NativeCall %s\n", n->as.native_call.name);
            break;
        case AST_FUNCTION_DEF:
            printf("FunctionDef %s\n", n->as.func_def.name);
            break;
        case AST_PARAMETER_LIST:
            printf("Parameter %s\n", n->as.param.param_name);
            break;
    }
    /* Print children */
    switch (n->type) {
        case AST_PROGRAM:
            print_ast(n->as.program.init, depth + 1);
            print_ast(n->as.program.main, depth + 1);
            print_ast(n->as.program.functions, depth + 1);
            break;
        case AST_BLOCK: {
            ASTNode *cur = n->as.block.stmts;
            while (cur) {
                print_ast(cur, depth + 1);
                cur = cur->next;
            }
            break;
        }
        case AST_UNARY:
            print_ast(n->as.unary.expr, depth + 1);
            break;
        case AST_BINARY:
            print_ast(n->as.binary.left, depth + 1);
            print_ast(n->as.binary.right, depth + 1);
            break;
        case AST_ASSIGNMENT:
            print_ast(n->as.assign.value, depth + 1);
            break;
        case AST_WHILE:
            print_ast(n->as.while_stmt.condition, depth + 1);
            print_ast(n->as.while_stmt.body, depth + 1);
            break;
        case AST_IF:
            print_ast(n->as.if_stmt.condition, depth + 1);
            print_ast(n->as.if_stmt.then_body, depth + 1);
            print_ast(n->as.if_stmt.else_body, depth + 1);
            break;
        case AST_FUNCTION_CALL: {
            ASTNode *arg = n->as.func_call.args;
            while (arg) {
                print_ast(arg, depth + 1);
                arg = arg->next;
            }
            break;
        }
        case AST_NATIVE_CALL: {
            ASTNode *arg = n->as.native_call.args;
            while (arg) {
                print_ast(arg, depth + 1);
                arg = arg->next;
            }
            break;
        }
        case AST_FUNCTION_DEF: {
            ASTNode *param = n->as.func_def.params;
            while (param) {
                print_ast(param, depth + 1);
                param = param->next;
            }
            print_ast(n->as.func_def.body, depth + 1);
            break;
        }
        case AST_PARAMETER_LIST:
            /* Parameters don't have children */
            break;
        default:
            break;
    }
    if (n->next && depth == 0) {
        print_ast(n->next, depth);
    }
}
int main(void) {
    const char *code =
        "init {\n"
        "  local x = 1 + 1;\n"
        "  local y = 2.1;\n"
        "  native print(x, y);\n"
        "  setup_environment();\n"
        "}\n"
        "main {\n"
        "  while (x < 10) { x = x + 1; }\n"
        "  if (x < 5) { x = x + 1; } elif (x < 8) { x = x + 2; } else { x = x - 1; }\n"
        "  local result = add(x, y);\n"
        "  process_data(x, y, result);\n"
        "  native print(result);\n"
        "  cleanup();\n"
        "}\n"
        "functions {\n"
        "  function add(a, b) {\n"
        "    local sum = a + b;\n"
        "    x = sum;\n"
        "  }\n"
        "  function setup_environment() {\n"
        "    native init_system();\n"
        "    local config = 42;\n"
        "  }\n"
        "  function process_data(x, y, result) {\n"
        "    native log(x, y, result);\n"
        "    if (result > 10) {\n"
        "      native alert(result);\n"
        "    }\n"
        "  }\n"
        "  function cleanup() {\n"
        "    native shutdown();\n"
        "  }\n"
        "}\n";

    Parser parser;
    Lexer lexer;
    lexer_init(&lexer, code);
    parser_init(&parser, &lexer);
    ASTNode *root = parse_program(&parser);
    print_ast(root, 0);
    ast_free(root);
    parser_free(&parser);
    lexer_free(&lexer);
    return 0;
}
#endif
