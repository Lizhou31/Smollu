#include <criterion/criterion.h>
#include <string.h>

#include "../src/components/compiler/smollu_compiler.h"

/* Helper assertions */
static void expect_node_type(ASTNode *node, NodeType t, const char *ctx) {
    cr_assert(node != NULL, "Node is NULL (%s)", ctx);
    cr_assert(node->type == t, "Expected node type %d, got %d (%s)", (int)t, (int)node->type, ctx);
}

static ASTNode *main_first_stmt(ASTNode *root) {
    cr_assert(root->type == AST_PROGRAM, "Root is not program");
    ASTNode *main_block = root->as.program.main;
    cr_assert(main_block && main_block->type == AST_BLOCK, "Main block missing");
    return main_block->as.block.stmts;
}

static ASTNode *init_first_stmt(ASTNode *root) {
    cr_assert(root->type == AST_PROGRAM, "Root is not program");
    ASTNode *init_block = root->as.program.init;
    cr_assert(init_block && init_block->type == AST_BLOCK, "Init block missing");
    return init_block->as.block.stmts;
}

Test(parser, precedence) {
    const char *code =
        "init {}\n"
        "main { 1 + 2 * 3; }";
    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    ASTNode *stmt = main_first_stmt(root);
    expect_node_type(stmt, AST_BINARY, "top binary");
    cr_assert_eq(stmt->as.binary.op, TOK_PLUS, "Top op should be +");

    ASTNode *left = stmt->as.binary.left;
    expect_node_type(left, AST_INT_LITERAL, "left operand");
    cr_assert_eq(left->as.int_val, 1);

    ASTNode *right = stmt->as.binary.right;
    expect_node_type(right, AST_BINARY, "right binary");
    cr_assert_eq(right->as.binary.op, TOK_STAR);

    ASTNode *r_left = right->as.binary.left;
    ASTNode *r_right = right->as.binary.right;
    expect_node_type(r_left, AST_INT_LITERAL, "nested left");
    cr_assert_eq(r_left->as.int_val, 2);
    expect_node_type(r_right, AST_INT_LITERAL, "nested right");
    cr_assert_eq(r_right->as.int_val, 3);

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
}

Test(parser, while_and_assignment) {
    const char *code =
        "init {\n"
        "  local x = 0;\n"
        "}\n"
        "main {\n"
        "  while (x < 5) { x = x + 1; }\n"
        "}";

    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    /* Expect two statements */
    ASTNode *init = root->as.program.init;
    ASTNode *main = root->as.program.main;
    cr_assert(init != NULL, "Init statement missing");
    cr_assert(main != NULL, "Main statement missing");

    /* First statement: local assignment */
    ASTNode *init_stmt = init_first_stmt(root);
    expect_node_type(init_stmt, AST_ASSIGNMENT, "init assignment");
    cr_assert_eq(init_stmt->as.assign.is_local, 1, "Expected local assignment");
    cr_assert_str_eq(init_stmt->as.assign.name, "x");

    /* Main has two statements: while loop and if statement */      
    ASTNode *main_stmt = main_first_stmt(root);
    expect_node_type(main_stmt, AST_WHILE, "while statement");

    /* Check while condition: x < 5 */
    ASTNode *cond = main_stmt->as.while_stmt.condition;
    expect_node_type(cond, AST_BINARY, "while condition");
    cr_assert_eq(cond->as.binary.op, TOK_LESS);

    /* Check while body exists */
    ASTNode *body = main_stmt->as.while_stmt.body;
    expect_node_type(body, AST_BLOCK, "while body");
    ASTNode *body_stmt = body->as.block.stmts;
    expect_node_type(body_stmt, AST_ASSIGNMENT, "while body assignment");
    cr_assert_str_eq(body_stmt->as.assign.name, "x");

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
} 

Test(parser, if_elif_else) {
    const char *code =
        "init {}\n"
        "main {\n"
        "  if (x < 3) { x = x + 1; }\n"
        "  elif (x < 6) { x = x + 2; }\n"
        "  else { x = x - 1; }\n"
        "}";

    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    ASTNode *if_stmt = main_first_stmt(root);
    expect_node_type(if_stmt, AST_IF, "if statement");

    /* Check condition: x < 3 */
    ASTNode *cond = if_stmt->as.if_stmt.condition;
    expect_node_type(cond, AST_BINARY, "if condition");
    cr_assert_eq(cond->as.binary.op, TOK_LESS);

    /* Check then body exists */
    ASTNode *then_body = if_stmt->as.if_stmt.then_body;
    expect_node_type(then_body, AST_BLOCK, "then body");

    /* Check else body is another if (elif) */
    ASTNode *else_body = if_stmt->as.if_stmt.else_body;
    expect_node_type(else_body, AST_IF, "elif as nested if");

    /* Check nested if condition: x < 6 */
    ASTNode *nested_cond = else_body->as.if_stmt.condition;
    expect_node_type(nested_cond, AST_BINARY, "nested if condition");
    cr_assert_eq(nested_cond->as.binary.op, TOK_LESS);

    /* Check nested if then body exists */
    ASTNode *nested_then_body = else_body->as.if_stmt.then_body;
    expect_node_type(nested_then_body, AST_BLOCK, "nested if then body");
    ASTNode *nested_then_body_stmt = nested_then_body->as.block.stmts;
    expect_node_type(nested_then_body_stmt, AST_ASSIGNMENT, "nested if then body assignment");
    cr_assert_str_eq(nested_then_body_stmt->as.assign.name, "x");

    /* Check nested if else body exists */
    ASTNode *nested_else_body = else_body->as.if_stmt.else_body;
    expect_node_type(nested_else_body, AST_BLOCK, "nested if else body");
    ASTNode *nested_else_body_stmt = nested_else_body->as.block.stmts;
    expect_node_type(nested_else_body_stmt, AST_ASSIGNMENT, "nested if else body assignment");
    cr_assert_str_eq(nested_else_body_stmt->as.assign.name, "x");

    /* Check else body is else block */
    ASTNode *else_body_stmt = else_body->as.if_stmt.else_body;
    expect_node_type(else_body_stmt, AST_BLOCK, "else body");
    ASTNode *else_body_stmt_stmt = else_body_stmt->as.block.stmts;
    expect_node_type(else_body_stmt_stmt, AST_ASSIGNMENT, "else body assignment");
    cr_assert_str_eq(else_body_stmt_stmt->as.assign.name, "x");

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
} 