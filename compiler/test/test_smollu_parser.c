#include <criterion/criterion.h>
#include <string.h>

#include "../smollu_compiler.h"

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
        "main { \n"
        "  1 + 2 * 3; \n"
        "  (1 + 2) * 3; \n"
        "  1 + 2 * 3 / 4; \n"
        "}"
        "functions {\n"
        "}";
    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    ASTNode *stmt = main_first_stmt(root);

    /* x = 1 + 2 * 3; */
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

    /* y = (1 + 2) * 3; */
    stmt = stmt->next;
    expect_node_type(stmt, AST_BINARY, "top binary");
    cr_assert_eq(stmt->as.binary.op, TOK_STAR, "Top op should be *");

    left = stmt->as.binary.left;
    expect_node_type(left, AST_BINARY, "left binary");
    cr_assert_eq(left->as.binary.op, TOK_PLUS, "Left op should be +");

    ASTNode *l_left = left->as.binary.left;
    ASTNode *l_right = left->as.binary.right;
    expect_node_type(l_left, AST_INT_LITERAL, "nested left");
    cr_assert_eq(l_left->as.int_val, 1);
    expect_node_type(l_right, AST_INT_LITERAL, "nested right");
    cr_assert_eq(l_right->as.int_val, 2);

    right = stmt->as.binary.right;
    expect_node_type(right, AST_INT_LITERAL, "right operand");
    cr_assert_eq(right->as.int_val, 3);


    /* z = 1 + 2 * 3 / 4; */
    stmt = stmt->next;
    expect_node_type(stmt, AST_BINARY, "top binary");
    cr_assert_eq(stmt->as.binary.op, TOK_PLUS, "Top op should be +");

    left = stmt->as.binary.left;
    expect_node_type(left, AST_INT_LITERAL, "left operand");
    cr_assert_eq(left->as.int_val, 1);

    right = stmt->as.binary.right;
    expect_node_type(right, AST_BINARY, "right binary");
    cr_assert_eq(right->as.binary.op, TOK_SLASH);

    r_right = right->as.binary.right;
    expect_node_type(r_right, AST_INT_LITERAL, "nested left");
    cr_assert_eq(r_right->as.int_val, 4);


    r_left = right->as.binary.left;
    expect_node_type(r_left, AST_BINARY, "nested right");
    cr_assert_eq(r_left->as.binary.op, TOK_STAR);

    ASTNode *r_l_left = r_left->as.binary.left;
    ASTNode *r_l_right = r_left->as.binary.right;
    expect_node_type(r_l_left, AST_INT_LITERAL, "nested left");
    cr_assert_eq(r_l_left->as.int_val, 2);
    expect_node_type(r_l_right, AST_INT_LITERAL, "nested right");
    cr_assert_eq(r_l_right->as.int_val, 3);

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
        "}"
        "functions {\n"
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
        "}"
        "functions {\n"
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

Test(parser, function_definition) {
    const char *code =
        "init {}\n"
        "main {\n"
        "  local result = add(1, 2);\n"
        "  local result = native print(1);\n"
        "}\n"
        "functions {\n"
        "  function add(a, b) {\n"
        "    local sum = a + b;\n"
        "    x = sum;\n"
        "    return sum;\n"
        "  }\n"
        "  function greet(name) {\n"
        "    native print(name);\n"
        "  }\n"
        "}";

    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    /* Check functions section */
    ASTNode *functions = root->as.program.functions;
    expect_node_type(functions, AST_BLOCK, "functions block");

    /* First function: add */
    ASTNode *func1 = functions->as.block.stmts;
    expect_node_type(func1, AST_FUNCTION_DEF, "first function def");
    cr_assert_str_eq(func1->as.func_def.name, "add");

    /* Check parameters */
    ASTNode *param1 = func1->as.func_def.params;
    expect_node_type(param1, AST_PARAMETER_LIST, "first parameter");
    cr_assert_str_eq(param1->as.param.param_name, "a");

    ASTNode *param2 = param1->next;
    expect_node_type(param2, AST_PARAMETER_LIST, "second parameter");
    cr_assert_str_eq(param2->as.param.param_name, "b");

    /* Check function body */
    ASTNode *body = func1->as.func_def.body;
    expect_node_type(body, AST_BLOCK, "function body");
    ASTNode *body_stmt = body->as.block.stmts;
    expect_node_type(body_stmt, AST_ASSIGNMENT, "function assignment");
    cr_assert_str_eq(body_stmt->as.assign.name, "sum");

    body_stmt = body_stmt->next;
    expect_node_type(body_stmt, AST_ASSIGNMENT, "function assignment");
    cr_assert_str_eq(body_stmt->as.assign.name, "x");

    body_stmt = body_stmt->next;
    expect_node_type(body_stmt, AST_RETURN, "function return");
    ASTNode *return_value = body_stmt->as.return_stmt.value;
    expect_node_type(return_value, AST_IDENTIFIER, "return value");
    cr_assert_str_eq(return_value->as.identifier, "sum");

    /* Second function: greet */
    ASTNode *func2 = func1->next;
    expect_node_type(func2, AST_FUNCTION_DEF, "second function def");
    cr_assert_str_eq(func2->as.func_def.name, "greet");

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
}

Test(parser, native_and_function_calls) {
    const char *code =
        "init {\n"
        "  native setup_gpio(pin, mode);\n"
        "  configure_system();\n"
        "}\n"
        "main {\n"
        "  local result = calculate(x, y);\n"
        "  native print(result);\n"
        "  process_data(result, 42);\n"
        "}\n"
        "functions {}";

    Lexer lex; lexer_init(&lex, code);
    Parser p; parser_init(&p, &lex);

    ASTNode *root = parse_program(&p);

    /* Check init section */
    ASTNode *init = root->as.program.init;
    ASTNode *init_stmt1 = init->as.block.stmts;

    /* First statement: native call */
    expect_node_type(init_stmt1, AST_NATIVE_CALL, "native call");
    cr_assert_str_eq(init_stmt1->as.native_call.name, "setup_gpio");

    /* Check arguments */
    ASTNode *arg1 = init_stmt1->as.native_call.args;
    expect_node_type(arg1, AST_IDENTIFIER, "first argument");
    ASTNode *arg2 = arg1->next;
    expect_node_type(arg2, AST_IDENTIFIER, "second argument");

    /* Second statement: function call */
    ASTNode *init_stmt2 = init_stmt1->next;
    expect_node_type(init_stmt2, AST_FUNCTION_CALL, "function call");
    cr_assert_str_eq(init_stmt2->as.func_call.name, "configure_system");

    /* Check main section */
    ASTNode *main = root->as.program.main;
    ASTNode *main_stmt1 = main->as.block.stmts;

    /* First statement: assignment with function call */
    expect_node_type(main_stmt1, AST_ASSIGNMENT, "assignment");
    ASTNode *assign_value = main_stmt1->as.assign.value;
    expect_node_type(assign_value, AST_FUNCTION_CALL, "function call in assignment");
    cr_assert_str_eq(assign_value->as.func_call.name, "calculate");

    /* Second statement: native call */
    ASTNode *main_stmt2 = main_stmt1->next;
    expect_node_type(main_stmt2, AST_NATIVE_CALL, "native call");
    cr_assert_str_eq(main_stmt2->as.native_call.name, "print");

    /* Third statement: function call */
    ASTNode *main_stmt3 = main_stmt2->next;
    expect_node_type(main_stmt3, AST_FUNCTION_CALL, "function call");
    cr_assert_str_eq(main_stmt3->as.func_call.name, "process_data");

    /* Cleanup */
    ast_free(root);
    parser_free(&p);
    lexer_free(&lex);
}