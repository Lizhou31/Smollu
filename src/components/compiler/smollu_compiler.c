/**
 * @file smollu_compiler.c
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Public API for the Smollu compiler
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
#include <errno.h>

#include "smollu_compiler.h"

typedef struct {
    const char *input_file;
    const char *output_file;
    int output_ast;
} Options;

static void usage(const char *prog_name) {
    fprintf(stderr,
        "Usage: %s [options] <source>\n"
        "Options:\n"
        "  -o <file>   Write output to <file>\n"
        "  -a          Output AST to file\n"
        "  -h          Show this help\n"
        "\n"
        "Examples:\n"
        "  %s foo.smol                     # output becomes foo.smolbc and foo.smolbc.ast (if -a is specified)\n"
        "  %s foo.smol -o build/foo.smolbc # custom path/name\n",
        prog_name, prog_name, prog_name);
}

static char *replace_text(const char *path, const char *new_ext) {
    const char *dot = strrchr(path, '.');
    size_t len = dot ? (size_t)(dot - path) : strlen(path);
    size_t ext_len = strlen(new_ext);
    char *out = malloc(len + ext_len + 1);
    if (!out) {
        fprintf(stderr, "Failed to allocate memory: %s\n", strerror(errno));
        exit(1);
    }
    memcpy(out, path, len);
    memcpy(out + len, new_ext, ext_len + 1);
    return out;
}

static int parse_args(int argc, char **argv, Options *opts) {
    memset(opts, 0, sizeof(*opts));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            if (++i == argc) { fprintf(stderr,"-o needs a file\n"); return -1; }
            opts->output_file = argv[i];
        } else if (strcmp(argv[i], "-a") == 0) {
            opts->output_ast = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help")==0) {
            usage(argv[0]);
            exit(0);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        } else {
            if (opts->input_file) {
                fprintf(stderr,"Only one source file allowed.\n");
                return -1;
            }
            opts->input_file = argv[i];
        } 
    }

    if (!opts->input_file) {
        fprintf(stderr, "No input file provided.\n");
        usage(argv[0]);
        return -1;
    }

    if (!opts->output_file) {
        opts->output_file = replace_text(opts->input_file, ".smolbc");
    }

    return 0;
}

static int open_files(const char *input_file, const char *output_file, int output_ast, FILE **in, FILE **out, FILE **ast_out) {
    *in = fopen(input_file, "r");
    if (!*in) {
        fprintf(stderr, "Failed to open input file: %s\n", input_file);
        return -1;
    }

    *out = fopen(output_file, "w+");
    if (!*out) {
        fprintf(stderr, "Failed to create output file: %s\n", output_file);
        return -1;
    }

    if (output_ast) {
        *ast_out = fopen(replace_text(output_file, ".smolbc.ast"), "w+");
        if (!*ast_out) {
            fprintf(stderr, "Failed to create AST output file: %s\n", replace_text(output_file, ".smolbc.ast"));
            return -1;
        }
    } else {
        *ast_out = NULL;
    }

    return 0;
}

static void print_indent(int n, FILE *out) { while (n--) fprintf(out, "  "); }

static void print_ast(ASTNode *n, int depth, FILE *out) {
    if (!n) return;
    print_indent(depth, out);
    switch (n->type) {
        case AST_PROGRAM:
            fprintf(out, "Program\n");
            break;
        case AST_BLOCK:
            fprintf(out, "Block\n");
            break;
        case AST_INT_LITERAL:
            fprintf(out, "Int %d\n", n->as.int_val);
            break;
        case AST_FLOAT_LITERAL:
            fprintf(out, "Float %f\n", n->as.float_val);
            break;
        case AST_BOOL_LITERAL:
            fprintf(out, "Bool %d\n", n->as.bool_val);
            break;
        case AST_NIL_LITERAL:
            fprintf(out, "Nil\n");
            break;
        case AST_IDENTIFIER:
            fprintf(out, "Identifier %s\n", n->as.identifier);
            break;
        case AST_UNARY:
            fprintf(out, "Unary %s\n", token_type_name(n->as.unary.op));
            break;
        case AST_BINARY:
            fprintf(out, "Binary %s\n", token_type_name(n->as.binary.op));
            break;
        case AST_ASSIGNMENT:
            fprintf(out, "Assign %s (local=%d)\n", n->as.assign.name, n->as.assign.is_local);
            break;
        case AST_WHILE:
            fprintf(out, "While\n");
            break;
        case AST_IF:
            fprintf(out, "If\n");
            break;
        case AST_FUNCTION_CALL:
            fprintf(out, "FunctionCall %s\n", n->as.func_call.name);
            break;
        case AST_NATIVE_CALL:
            fprintf(out, "NativeCall %s\n", n->as.native_call.name);
            break;
        case AST_FUNCTION_DEF:
            fprintf(out, "FunctionDef %s\n", n->as.func_def.name);
            break;
        case AST_PARAMETER_LIST:
            fprintf(out, "Parameter %s\n", n->as.param.param_name);
            break;
    }
    /* Print children */
    switch (n->type) {
        case AST_PROGRAM:
            print_ast(n->as.program.init, depth + 1, out);
            print_ast(n->as.program.main, depth + 1, out);
            print_ast(n->as.program.functions, depth + 1, out);
            break;
        case AST_BLOCK: {
            ASTNode *cur = n->as.block.stmts;
            while (cur) {
                print_ast(cur, depth + 1, out);
                cur = cur->next;
            }
            break;
        }
        case AST_UNARY:
            print_ast(n->as.unary.expr, depth + 1, out);
            break;
        case AST_BINARY:
            print_ast(n->as.binary.left, depth + 1, out);
            print_ast(n->as.binary.right, depth + 1, out);
            break;
        case AST_ASSIGNMENT:
            print_ast(n->as.assign.value, depth + 1, out);
            break;
        case AST_WHILE:
            print_ast(n->as.while_stmt.condition, depth + 1, out);
            print_ast(n->as.while_stmt.body, depth + 1, out);
            break;
        case AST_IF:
            print_ast(n->as.if_stmt.condition, depth + 1, out);
            print_ast(n->as.if_stmt.then_body, depth + 1, out);
            print_ast(n->as.if_stmt.else_body, depth + 1, out);
            break;
        case AST_FUNCTION_CALL:
            print_ast(n->as.func_call.args, depth + 1, out);
            break;
        case AST_NATIVE_CALL:
            print_ast(n->as.native_call.args, depth + 1, out);
            break;
        case AST_FUNCTION_DEF:
            print_ast(n->as.func_def.params, depth + 1, out);
            print_ast(n->as.func_def.body, depth + 1, out);
            break;
        case AST_PARAMETER_LIST:
            /* Parameters don't have children */
            break;
        default:
            break;
    }
    if (n->next && depth == 0) {
        print_ast(n->next, depth, out);
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Public Compiler API                                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

int smollu_compile(FILE *in, FILE *out, FILE *ast_out) {

    /* Read entire input file into memory */
    fseek(in, 0, SEEK_END);
    long file_size = ftell(in);
    rewind(in);

    char *src = (char *)malloc(file_size + 1);
    if (!src) {
        fprintf(stderr, "Failed to allocate memory for source code\n");
        return -1;
    }

    size_t bytes_read = fread(src, 1, file_size, in);
    if (bytes_read < (size_t)file_size) {
        fprintf(stderr, "Failed to read entire input file\n");
        free(src);
        return -1;
    }
    src[file_size] = '\0';

    Lexer lex;
    Parser parser;
    lexer_init(&lex, src);
    parser_init(&parser, &lex);
    ASTNode *root = parse_program(&parser);

    /* Print bytecode to output file */
    if (out) {
        // print_ast(root, 0, out);
        // fflush(out);
    }

    /* Print AST to output file */
    if (ast_out) {
        print_ast(root, 0, ast_out);
        fflush(ast_out);
    }

    ast_free(root);
    parser_free(&parser);
    lexer_free(&lex);
    free(src);

    return 0;
}

int main(int argc, char **argv) {
    Options opts;
    if (parse_args(argc, argv, &opts) < 0) {
        return 1;
    }

    FILE *in, *out, *ast_out;
    if (open_files(opts.input_file, opts.output_file, opts.output_ast, &in, &out, &ast_out) < 0) {
        return 1;
    }

    printf("Compiling %s to %s\n", opts.input_file, opts.output_file);
    smollu_compile(in, out, ast_out);

    fclose(in);
    fclose(out);
    if (opts.output_ast) {
        fclose(ast_out);
    }
    return 0;
}
