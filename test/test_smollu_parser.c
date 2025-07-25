#include <criterion/criterion.h>
#include <string.h>

#include "../src/components/compiler/smollu_compiler.h"

/* Helper assertions */
static void expect_node_type(ASTNode *node, NodeType t, const char *ctx) {
    cr_assert(node != NULL, "Node is NULL (%s)", ctx);
    cr_assert(node->type == t, "Expected node type %d, got %d (%s)", (int)t, (int)node->type, ctx);
}

static ASTNode *first_stmt(ASTNode *root) {
    cr_assert(root->type == AST_PROGRAM, "Root is not program");
    return root->as.block.stmts;
}

Test(parser, precedence) {
    const char *code = "1 + 2 * 3;";
    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    ASTNode *stmt = first_stmt(root);
    expect_node_type(stmt, AST_BINARY, "top binary");
    cr_assert_eq(stmt->as.binary.op, TOK_PLUS, "Top op should be +");

    /* left operand: 1 */
    ASTNode *left = stmt->as.binary.left;
    expect_node_type(left, AST_INT_LITERAL, "left operand");
    cr_assert_eq(left->as.int_val, 1);

    /* right operand: 2 * 3 */
    ASTNode *right = stmt->as.binary.right;
    expect_node_type(right, AST_BINARY, "right binary");
    cr_assert_eq(right->as.binary.op, TOK_STAR, "Nested op should be *");

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
        "local x = 0;"
        "while (x < 5) { x = x + 1; }";

    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    /* Expect two statements */
    ASTNode *first = first_stmt(root);
    ASTNode *second = first->next;
    cr_assert(second != NULL, "Second statement missing");
    cr_assert(second->next == NULL, "Unexpected extra statements");

    /* First statement: local assignment */
    expect_node_type(first, AST_ASSIGNMENT, "first statement");
    cr_assert_eq(first->as.assign.is_local, 1, "Expected local assignment");
    cr_assert_str_eq(first->as.assign.name, "x");

    /* Second statement: while loop */
    expect_node_type(second, AST_WHILE, "while statement");

    /* Condition should be x < 5 */
    ASTNode *cond = second->as.while_stmt.condition;
    expect_node_type(cond, AST_BINARY, "while condition");
    cr_assert_eq(cond->as.binary.op, TOK_LESS);

    /* Body should be block containing assignment */
    ASTNode *body = second->as.while_stmt.body;
    expect_node_type(body, AST_BLOCK, "while body");
    ASTNode *body_stmt = body->as.block.stmts;
    expect_node_type(body_stmt, AST_ASSIGNMENT, "body assignment");
    cr_assert_str_eq(body_stmt->as.assign.name, "x");

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
} 