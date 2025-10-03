//===- SmolMLIRGen.cpp - MLIR generator for Smol dialect -------*- C++ -*-===//
//
// Generates high-level Smol dialect MLIR from Smollu AST
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolMLIRGen.h"
#include "Smollu/SmolOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <iostream>

using namespace mlir;
using namespace mlir::smol;
using SmolluASTNode = mlir::smollu::SmolluASTNode;

SmolMLIRGenerator::SmolMLIRGenerator(MLIRContext *ctx)
    : context(ctx), builder(ctx) {
    builder.getContext()->loadDialect<SmolDialect>();
    builder.getContext()->loadDialect<mlir::func::FuncDialect>();
    // Use unknown location for module (no specific source location)
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module.getBody());
}

ModuleOp SmolMLIRGenerator::generateMLIR(const SmolluASTNode &ast) {
    std::cout << "=== Smol MLIR Generator: Starting generation ===\n";

    // Process all top-level blocks
    for (const auto &child : ast.children) {
        if (child.type == "InitBlock") {
            generateInitBlock(child);
        } else if (child.type == "MainBlock") {
            generateMainBlock(child);
        } else if (child.type == "FunctionsBlock") {
            generateFunctionsBlock(child);
        }
    }

    std::cout << "=== Smol MLIR Generation completed ===\n";
    return module;
}

void SmolMLIRGenerator::generateMainBlock(const SmolluASTNode &mainNode) {
    clearLocalVars();

    // Create main function using func dialect
    auto funcType = builder.getFunctionType({}, {});
    auto mainFunc = builder.create<mlir::func::FuncOp>(
        getLoc(mainNode), "main", funcType);

    Block *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate statements
    for (const auto &stmt : mainNode.children) {
        generateStatement(stmt);
    }

    // Add return
    builder.create<mlir::func::ReturnOp>(getLoc(mainNode));

    // Restore insertion point to module level
    builder.setInsertionPointToEnd(module.getBody());
}

void SmolMLIRGenerator::generateStatement(const SmolluASTNode &stmt) {
    if (stmt.type == "Assignment") {
        generateAssignment(stmt, false);
    } else if (stmt.type == "LocalAssignment") {
        generateAssignment(stmt, true);
    } else if (stmt.type == "WhileStatement") {
        generateWhileStatement(stmt);
    } else if (stmt.type == "IfStatement") {
        generateIfStatement(stmt);
    } else if (stmt.type == "NativeCall") {
        generateNativeCall(stmt);
    } else if (stmt.type == "ReturnStatement") {
        generateReturnStatement(stmt);
    } else if (stmt.type == "FunctionCall") {
        generateFunctionCall(stmt);
    } else {
        // Expression statement
        generateExpression(stmt);
    }
}

void SmolMLIRGenerator::generateAssignment(const SmolluASTNode &assign, bool isLocal) {
    std::string varName = assign.value;

    if (assign.children.empty()) {
        std::cerr << "Error: Assignment without expression\n";
        return;
    }

    // Track variable scope
    std::string scope = isLocal ? "local" : "global";
    variableScopes[varName] = scope;

    // Generate RHS expression
    Value rhs = generateExpression(assign.children[0]);
    if (rhs) {
        // Use smol.var_store operation
        builder.create<VarStoreOp>(
            getLoc(assign),
            builder.getStringAttr(varName),
            rhs
        );
    }
}

Value SmolMLIRGenerator::generateExpression(const SmolluASTNode &expr) {
    if (expr.type == "IntLiteral") {
        int32_t val = std::stoi(expr.value);
        OperationState state(getLoc(expr), ConstantIntOp::getOperationName());
        ConstantIntOp::build(builder, state, builder.getI32Type(), builder.getI32IntegerAttr(val));
        return builder.create(state)->getResult(0);
    } else if (expr.type == "FloatLiteral") {
        float val = std::stof(expr.value);
        OperationState state(getLoc(expr), ConstantFloatOp::getOperationName());
        ConstantFloatOp::build(builder, state, builder.getF32Type(), builder.getF32FloatAttr(val));
        return builder.create(state)->getResult(0);
    } else if (expr.type == "BoolLiteral") {
        bool val = (expr.value == "true");
        OperationState state(getLoc(expr), ConstantBoolOp::getOperationName());
        ConstantBoolOp::build(builder, state, builder.getI1Type(), builder.getBoolAttr(val));
        return builder.create(state)->getResult(0);
    } else if (expr.type == "Identifier") {
        // Load variable using smol.var_load
        std::string varName = expr.value;

        // Determine type based on context (default to i32)
        Type resultType = builder.getI32Type();

        return builder.create<VarLoadOp>(
            getLoc(expr),
            resultType,
            builder.getStringAttr(varName)
        ).getResult();
    } else if (expr.type == "BinaryOp" && expr.children.size() == 2) {
        Value left = generateExpression(expr.children[0]);
        Value right = generateExpression(expr.children[1]);
        if (!left || !right) return nullptr;

        Type resultType = left.getType();
        // If types differ and one is float, promote to float
        if (left.getType() != right.getType()) {
            if (llvm::isa<FloatType>(left.getType()) || llvm::isa<FloatType>(right.getType())) {
                resultType = builder.getF32Type();
            }
        }

        // Arithmetic operators
        if (expr.value == "+") {
            return builder.create<AddOp>(getLoc(expr), resultType, left, right).getResult();
        } else if (expr.value == "-") {
            return builder.create<SubOp>(getLoc(expr), resultType, left, right).getResult();
        } else if (expr.value == "*") {
            return builder.create<MulOp>(getLoc(expr), resultType, left, right).getResult();
        } else if (expr.value == "/") {
            return builder.create<DivOp>(getLoc(expr), resultType, left, right).getResult();
        } else if (expr.value == "%") {
            return builder.create<ModOp>(getLoc(expr), resultType, left, right).getResult();
        }
        // Comparison operators
        else if (expr.value == "<") {
            return builder.create<LtOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == "<=") {
            return builder.create<LeOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == ">") {
            return builder.create<GtOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == ">=") {
            return builder.create<GeOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == "==") {
            return builder.create<EqOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == "!=") {
            return builder.create<NeOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        }
        // Logical operators
        else if (expr.value == "&&") {
            return builder.create<AndOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        } else if (expr.value == "||") {
            return builder.create<OrOp>(getLoc(expr), builder.getI1Type(), left, right).getResult();
        }
    } else if (expr.type == "UnaryOp" && expr.children.size() == 1) {
        Value operand = generateExpression(expr.children[0]);
        if (!operand) return nullptr;

        if (expr.value == "-") {
            return builder.create<NegOp>(getLoc(expr), operand.getType(), operand).getResult();
        } else if (expr.value == "!") {
            return builder.create<NotOp>(getLoc(expr), builder.getI1Type(), operand).getResult();
        }
    } else if (expr.type == "NativeCall") {
        // Handle native calls in expression context
        std::vector<Value> args;
        for (const auto &arg : expr.children) {
            Value argValue = generateExpression(arg);
            if (argValue) {
                args.push_back(argValue);
            }
        }

        // Create native call operation with return value
        auto nativeCallOp = builder.create<NativeCallOp>(
            getLoc(expr),
            builder.getI32Type(),  // Default return type
            builder.getStringAttr(expr.value),
            args
        );
        return nativeCallOp.getResult();
    } else if (expr.type == "FunctionCall") {
        return generateFunctionCall(expr);
    }
    return nullptr;
}

void SmolMLIRGenerator::generateWhileStatement(const SmolluASTNode &whileStmt) {
    if (whileStmt.children.size() < 2) {
        std::cerr << "Error: While statement missing condition or body\n";
        return;
    }

    auto whileOp = builder.create<WhileOp>(getLoc(whileStmt));

    // Create condition region
    Region &condRegion = whileOp.getCondition();
    Block *condBlock = builder.createBlock(&condRegion);
    builder.setInsertionPointToStart(condBlock);

    // Generate condition expression
    Value condValue = generateExpression(whileStmt.children[0]);
    if (condValue) {
        OperationState state(getLoc(whileStmt.children[0]), YieldOp::getOperationName());
        YieldOp::build(builder, state, ValueRange{condValue});
        builder.create(state);
    }

    // Create body region
    Region &bodyRegion = whileOp.getBody();
    Block *bodyBlock = builder.createBlock(&bodyRegion);
    builder.setInsertionPointToStart(bodyBlock);

    // Generate body statements
    generateBlock(whileStmt.children[1]);

    // Add yield operation to terminate body
    OperationState yieldState(getLoc(whileStmt.children[1]), YieldOp::getOperationName());
    YieldOp::build(builder, yieldState, ValueRange{});
    builder.create(yieldState);

    // Restore insertion point after while
    builder.setInsertionPointAfter(whileOp);
}

void SmolMLIRGenerator::generateIfStatement(const SmolluASTNode &ifStmt) {
    if (ifStmt.children.size() < 2) {
        std::cerr << "Error: If statement missing condition or body\n";
        return;
    }

    Value condValue = generateExpression(ifStmt.children[0]);
    if (!condValue) return;

    auto ifOp = builder.create<IfOp>(getLoc(ifStmt), condValue);

    // Create then region
    Region &thenRegion = ifOp.getThenRegion();
    Block *thenBlock = builder.createBlock(&thenRegion);
    builder.setInsertionPointToStart(thenBlock);

    // Generate then body
    generateBlock(ifStmt.children[1]);
    OperationState yieldState(getLoc(ifStmt.children[1]), YieldOp::getOperationName());
    YieldOp::build(builder, yieldState, ValueRange{});
    builder.create(yieldState);

    // Create else region if present
    if (ifStmt.children.size() > 2) {
        Region &elseRegion = ifOp.getElseRegion();
        Block *elseBlock = builder.createBlock(&elseRegion);
        builder.setInsertionPointToStart(elseBlock);

        const auto &elseChild = ifStmt.children[2];
        if (elseChild.type == "IfStatement") {
            // Nested elif
            generateIfStatement(elseChild);
        } else {
            // Regular else block
            generateBlock(elseChild);
        }
        OperationState yieldState(getLoc(elseChild), YieldOp::getOperationName());
        YieldOp::build(builder, yieldState, ValueRange{});
        builder.create(yieldState);
    }

    // Restore insertion point after if
    builder.setInsertionPointAfter(ifOp);
}

void SmolMLIRGenerator::generateBlock(const SmolluASTNode &block) {
    for (const auto &stmt : block.children) {
        generateStatement(stmt);
    }
}

void SmolMLIRGenerator::generateNativeCall(const SmolluASTNode &nativeCall) {
    std::vector<Value> args;
    for (const auto &child : nativeCall.children) {
        Value arg = generateExpression(child);
        if (arg) args.push_back(arg);
    }

    // Create native call operation without return value
    builder.create<NativeCallOp>(
        getLoc(nativeCall),
        TypeRange{},  // No return value in statement context
        builder.getStringAttr(nativeCall.value),
        args
    );
}

void SmolMLIRGenerator::clearLocalVars() {
    // Remove all local variables from scope tracking
    for (auto it = variableScopes.begin(); it != variableScopes.end(); ) {
        if (it->second == "local") {
            it = variableScopes.erase(it);
        } else {
            ++it;
        }
    }
}

std::string SmolMLIRGenerator::getVarScope(const std::string &name) {
    auto it = variableScopes.find(name);
    if (it != variableScopes.end()) {
        return it->second;
    }
    return "global"; // Default to global
}

void SmolMLIRGenerator::generateInitBlock(const SmolluASTNode &initNode) {
    clearLocalVars();

    auto funcType = builder.getFunctionType({}, {});
    auto initFunc = builder.create<mlir::func::FuncOp>(
        getLoc(initNode), "__init", funcType);

    Block *entryBlock = initFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    for (const auto &stmt : initNode.children) {
        generateStatement(stmt);
    }

    builder.create<mlir::func::ReturnOp>(getLoc(initNode));
    builder.setInsertionPointToEnd(module.getBody());
}

void SmolMLIRGenerator::generateFunctionsBlock(const SmolluASTNode &functionsNode) {
    for (const auto &child : functionsNode.children) {
        if (child.type == "FunctionDefinition") {
            generateFunctionDefinition(child);
        }
    }
}

void SmolMLIRGenerator::generateFunctionDefinition(const SmolluASTNode &funcDef) {
    std::string funcName = funcDef.value;
    clearLocalVars();

    // Parse parameters
    std::vector<Type> paramTypes;
    size_t bodyIndex = 0;

    for (size_t i = 0; i < funcDef.children.size(); ++i) {
        if (funcDef.children[i].type == "Block") {
            bodyIndex = i;
            break;
        } else if (funcDef.children[i].type == "Identifier") {
            // Register parameter as local variable
            std::string paramName = funcDef.children[i].value;
            variableScopes[paramName] = "local";
            paramTypes.push_back(builder.getI32Type());
        }
    }

    // Create function
    auto funcType = builder.getFunctionType(paramTypes, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        getLoc(funcDef), funcName, funcType);

    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Store function arguments into their corresponding variables
    size_t argIndex = 0;
    for (size_t i = 0; i < funcDef.children.size(); ++i) {
        if (funcDef.children[i].type == "Identifier" && i < bodyIndex) {
            std::string paramName = funcDef.children[i].value;
            Value arg = entryBlock->getArgument(argIndex++);
            builder.create<VarStoreOp>(
                getLoc(funcDef.children[i]),
                builder.getStringAttr(paramName),
                arg
            );
        }
    }

    // Generate function body
    if (bodyIndex < funcDef.children.size()) {
        generateBlock(funcDef.children[bodyIndex]);
    }

    // Add default return
    builder.create<mlir::func::ReturnOp>(getLoc(funcDef));
    builder.setInsertionPointToEnd(module.getBody());
}

void SmolMLIRGenerator::generateReturnStatement(const SmolluASTNode &returnStmt) {
    if (returnStmt.children.empty()) {
        builder.create<ReturnOp>(getLoc(returnStmt), Value());
    } else {
        Value retValue = generateExpression(returnStmt.children[0]);
        if (retValue) {
            builder.create<ReturnOp>(getLoc(returnStmt), retValue);
        } else {
            builder.create<ReturnOp>(getLoc(returnStmt), Value());
        }
    }
}

Value SmolMLIRGenerator::generateFunctionCall(const SmolluASTNode &funcCall) {
    std::string funcName = funcCall.value;

    // Generate arguments
    std::vector<Value> args;
    for (const auto &arg : funcCall.children) {
        Value argValue = generateExpression(arg);
        if (argValue) {
            args.push_back(argValue);
        }
    }

    // Create call operation
    auto callOp = builder.create<CallOp>(
        getLoc(funcCall),
        builder.getI32Type(),
        mlir::SymbolRefAttr::get(builder.getContext(), funcName),
        args
    );

    return callOp.getResult();
}
