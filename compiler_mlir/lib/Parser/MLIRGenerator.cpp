//===- MLIRGenerator.cpp - MLIR generator from AST -------------*- C++ -*-===//
//
// MLIR generator that converts Smollu AST to MLIR representation
//
//===----------------------------------------------------------------------===//

#include "Smollu/SmolluMLIRGen.h"
#include <iostream>

using namespace mlir;
using namespace mlir::smollu;
using SmolluASTNode = mlir::smollu::SmolluASTNode;

SmolluMLIRGenerator::SmolluMLIRGenerator(MLIRContext *ctx)
    : context(ctx), builder(ctx) {
    builder.getContext()->loadDialect<SmolluDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
}

ModuleOp SmolluMLIRGenerator::generateMLIR(const SmolluASTNode &ast) {
    std::cout << "=== MLIR Generator: Starting generation ===\n";

    // Find and process main block
    for (const auto &child : ast.children) {
        if (child.type == "MainBlock") {
            generateMainBlock(child);
            break;
        }
    }

    std::cout << "=== MLIR Generation completed ===\n";
    return module;
}

void SmolluMLIRGenerator::generateMainBlock(const SmolluASTNode &mainNode) {
    // Create MLIR main function
    auto funcType = builder.getFunctionType({}, {});
    auto mainFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);

    Block *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate statements
    for (const auto &stmt : mainNode.children) {
        generateStatement(stmt);
    }

    // Add return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    module.push_back(mainFunc);
}

void SmolluMLIRGenerator::generateStatement(const SmolluASTNode &stmt) {
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
    } else {
        // Expression statement
        generateExpression(stmt);
    }
}

void SmolluMLIRGenerator::generateAssignment(const SmolluASTNode &assign, bool isLocal) {
    std::string varName = assign.value;

    if (assign.children.empty()) {
        std::cerr << "Error: Assignment without expression\n";
        return;
    }

    Value rhs = generateExpression(assign.children[0]);
    if (rhs) {
        if (isLocal) {
            uint8_t slot = getOrCreateLocalVar(varName);
            builder.create<SetLocalOp>(builder.getUnknownLoc(),
                builder.getI8IntegerAttr(slot), rhs);
        } else {
            uint8_t slot = getOrCreateGlobalVar(varName);
            builder.create<SetGlobalOp>(builder.getUnknownLoc(),
                builder.getI8IntegerAttr(slot), rhs);
        }
    }
}

Value SmolluMLIRGenerator::generateExpression(const SmolluASTNode &expr) {
    if (expr.type == "IntLiteral") {
        int32_t val = std::stoi(expr.value);
        return builder.create<ConstantIntOp>(builder.getUnknownLoc(),
            builder.getI32Type(), builder.getI32IntegerAttr(val));
    } else if (expr.type == "FloatLiteral") {
        float val = std::stof(expr.value);
        return builder.create<ConstantFloatOp>(builder.getUnknownLoc(),
            builder.getF32Type(), builder.getF32FloatAttr(val));
    } else if (expr.type == "BoolLiteral") {
        bool val = (expr.value == "true");
        return builder.create<ConstantBoolOp>(builder.getUnknownLoc(),
            builder.getI1Type(), builder.getBoolAttr(val));
    } else if (expr.type == "Identifier") {
        // Try local first, then global
        auto localIt = localVars.find(expr.value);
        if (localIt != localVars.end()) {
            return builder.create<GetLocalOp>(builder.getUnknownLoc(),
                builder.getI32Type(), builder.getI8IntegerAttr(localIt->second));
        }

        uint8_t slot = getOrCreateGlobalVar(expr.value);
        return builder.create<GetGlobalOp>(builder.getUnknownLoc(),
            builder.getI32Type(), builder.getI8IntegerAttr(slot));
    } else if (expr.type == "BinaryOp" && expr.children.size() == 2) {
        Value left = generateExpression(expr.children[0]);
        Value right = generateExpression(expr.children[1]);
        if (!left || !right) return nullptr;

        if (expr.value == "TOK_PLUS") {
            return builder.create<AddOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "TOK_MINUS") {
            return builder.create<SubOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "TOK_STAR") {
            return builder.create<MulOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "TOK_SLASH") {
            return builder.create<DivOp>(builder.getUnknownLoc(), left.getType(), left, right);
        }
        // TODO: Add more operators
    } else if (expr.type == "UnaryOp" && expr.children.size() == 1) {
        Value operand = generateExpression(expr.children[0]);
        if (!operand) return nullptr;

        if (expr.value == "TOK_MINUS") {
            // Create negative constant and subtract
            Value zero = builder.create<ConstantIntOp>(builder.getUnknownLoc(),
                operand.getType(), builder.getIntegerAttr(operand.getType(), 0));
            return builder.create<SubOp>(builder.getUnknownLoc(), operand.getType(), zero, operand);
        } else if (expr.value == "TOK_BANG") {
            // TODO: Add logical not operation
        }
    } else if (expr.type == "NativeCall") {
        // Handle native calls that return values
        generateNativeCall(expr);
        // TODO: Return appropriate value for native functions that return values
    }
    return nullptr;
}

void SmolluMLIRGenerator::generateWhileStatement(const SmolluASTNode &whileStmt) {
    if (whileStmt.children.size() < 2) {
        std::cerr << "Error: While statement missing condition or body\n";
        return;
    }

    auto whileOp = builder.create<WhileOp>(builder.getUnknownLoc());
    // TODO: Implement condition and body regions
    // This requires more complex region handling
}

void SmolluMLIRGenerator::generateIfStatement(const SmolluASTNode &ifStmt) {
    if (ifStmt.children.size() < 2) {
        std::cerr << "Error: If statement missing condition or body\n";
        return;
    }

    Value condValue = generateExpression(ifStmt.children[0]);
    if (condValue) {
        auto ifOp = builder.create<IfOp>(builder.getUnknownLoc(), condValue);
        // TODO: Implement then/else regions
        // This requires more complex region handling
    }
}

void SmolluMLIRGenerator::generateBlock(const SmolluASTNode &block) {
    for (const auto &stmt : block.children) {
        generateStatement(stmt);
    }
}

void SmolluMLIRGenerator::generateNativeCall(const SmolluASTNode &nativeCall) {
    if (nativeCall.value == "print") {
        std::vector<Value> args;
        for (const auto &child : nativeCall.children) {
            Value arg = generateExpression(child);
            if (arg) args.push_back(arg);
        }
        builder.create<PrintOp>(builder.getUnknownLoc(), args);
    }
    // TODO: Add support for other native functions
}

uint8_t SmolluMLIRGenerator::getOrCreateGlobalVar(const std::string &name) {
    auto it = globalVars.find(name);
    if (it != globalVars.end()) {
        return it->second;
    }

    uint8_t slot = nextGlobalSlot++;
    globalVars[name] = slot;
    return slot;
}

uint8_t SmolluMLIRGenerator::getOrCreateLocalVar(const std::string &name) {
    auto it = localVars.find(name);
    if (it != localVars.end()) {
        return it->second;
    }

    uint8_t slot = nextLocalSlot++;
    localVars[name] = slot;
    return slot;
}