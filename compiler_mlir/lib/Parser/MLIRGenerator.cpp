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
    // Set insertion point to the module body
    builder.setInsertionPointToEnd(module.getBody());
}

ModuleOp SmolluMLIRGenerator::generateMLIR(const SmolluASTNode &ast) {
    std::cout << "=== MLIR Generator: Starting generation ===\n";

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

    std::cout << "=== MLIR Generation completed ===\n";
    return module;
}

void SmolluMLIRGenerator::generateMainBlock(const SmolluASTNode &mainNode) {
    // Clear local variables for main function scope
    clearLocalVars();

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

    // Restore insertion point to module level
    builder.setInsertionPointToEnd(module.getBody());
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
    } else if (stmt.type == "ReturnStatement") {
        generateReturnStatement(stmt);
    } else if (stmt.type == "FunctionCall") {
        generateFunctionCall(stmt);
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

        // Arithmetic operators
        if (expr.value == "+") {
            return builder.create<AddOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "-") {
            return builder.create<SubOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "*") {
            return builder.create<MulOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "/") {
            return builder.create<DivOp>(builder.getUnknownLoc(), left.getType(), left, right);
        } else if (expr.value == "%") {
            return builder.create<ModOp>(builder.getUnknownLoc(), left.getType(), left, right);
        }
        // Comparison operators
        else if (expr.value == "<") {
            return builder.create<LtOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == "<=") {
            return builder.create<LeOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == ">") {
            return builder.create<GtOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == ">=") {
            return builder.create<GeOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == "==") {
            return builder.create<EqOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == "!=") {
            return builder.create<NeOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        }
        // Logical operators
        else if (expr.value == "&&") {
            return builder.create<AndOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        } else if (expr.value == "||") {
            return builder.create<OrOp>(builder.getUnknownLoc(), builder.getI1Type(), left, right);
        }
    } else if (expr.type == "UnaryOp" && expr.children.size() == 1) {
        Value operand = generateExpression(expr.children[0]);
        if (!operand) return nullptr;

        if (expr.value == "-") {
            // Create negative constant and subtract
            Value zero = builder.create<ConstantIntOp>(builder.getUnknownLoc(),
                operand.getType(), builder.getIntegerAttr(operand.getType(), 0));
            return builder.create<SubOp>(builder.getUnknownLoc(), operand.getType(), zero, operand);
        } else if (expr.value == "!") {
            // Logical NOT operation
            return builder.create<NotOp>(builder.getUnknownLoc(), builder.getI1Type(), operand);
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
            builder.getUnknownLoc(),
            builder.getI32Type(),  // Default return type
            builder.getStringAttr(expr.value),
            args
        );
        return nativeCallOp.getResult();
    } else if (expr.type == "FunctionCall") {
        // Handle function calls
        return generateFunctionCall(expr);
    }
    return nullptr;
}

void SmolluMLIRGenerator::generateWhileStatement(const SmolluASTNode &whileStmt) {
    if (whileStmt.children.size() < 2) {
        std::cerr << "Error: While statement missing condition or body\n";
        return;
    }

    auto whileOp = builder.create<WhileOp>(builder.getUnknownLoc());

    // Create condition region
    Region &condRegion = whileOp.getCondition();
    Block *condBlock = builder.createBlock(&condRegion);
    builder.setInsertionPointToStart(condBlock);

    // Generate condition expression
    Value condValue = generateExpression(whileStmt.children[0]);
    if (condValue) {
        // Add yield operation to return condition value
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), ValueRange{condValue});
    }

    // Create body region
    Region &bodyRegion = whileOp.getBody();
    Block *bodyBlock = builder.createBlock(&bodyRegion);
    builder.setInsertionPointToStart(bodyBlock);

    // Generate body statements
    generateBlock(whileStmt.children[1]);

    // Add yield operation to terminate body
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    // Restore insertion point after while
    builder.setInsertionPointAfter(whileOp);
}

void SmolluMLIRGenerator::generateIfStatement(const SmolluASTNode &ifStmt) {
    if (ifStmt.children.size() < 2) {
        std::cerr << "Error: If statement missing condition or body\n";
        return;
    }

    Value condValue = generateExpression(ifStmt.children[0]);
    if (!condValue) return;

    auto ifOp = builder.create<IfOp>(builder.getUnknownLoc(), condValue);

    // Create then region
    Region &thenRegion = ifOp.getThenRegion();
    Block *thenBlock = builder.createBlock(&thenRegion);
    builder.setInsertionPointToStart(thenBlock);

    // Generate then body
    generateBlock(ifStmt.children[1]);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    // Create else region if present (child index 2 or later)
    if (ifStmt.children.size() > 2) {
        Region &elseRegion = ifOp.getElseRegion();
        Block *elseBlock = builder.createBlock(&elseRegion);
        builder.setInsertionPointToStart(elseBlock);

        // Check if it's an elif (nested IfStatement) or else block
        const auto &elseChild = ifStmt.children[2];
        if (elseChild.type == "IfStatement") {
            // Nested elif - generate as another if statement
            generateIfStatement(elseChild);
        } else {
            // Regular else block
            generateBlock(elseChild);
        }
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    }

    // Restore insertion point after if
    builder.setInsertionPointAfter(ifOp);
}

void SmolluMLIRGenerator::generateBlock(const SmolluASTNode &block) {
    for (const auto &stmt : block.children) {
        generateStatement(stmt);
    }
}

void SmolluMLIRGenerator::generateNativeCall(const SmolluASTNode &nativeCall) {
    std::vector<Value> args;
    for (const auto &child : nativeCall.children) {
        Value arg = generateExpression(child);
        if (arg) args.push_back(arg);
    }

    // Use specific operations for known native functions
    if (nativeCall.value == "print") {
        builder.create<PrintOp>(builder.getUnknownLoc(), args);
    } else {
        // Generic native call for other functions
        builder.create<NativeCallOp>(
            builder.getUnknownLoc(),
            TypeRange{},  // No return value in statement context
            builder.getStringAttr(nativeCall.value),
            args
        );
    }
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

void SmolluMLIRGenerator::clearLocalVars() {
    localVars.clear();
    nextLocalSlot = 0;
}

void SmolluMLIRGenerator::generateInitBlock(const SmolluASTNode &initNode) {
    // Clear local variables for init function scope
    clearLocalVars();

    // Create MLIR init function (called before main)
    auto funcType = builder.getFunctionType({}, {});
    auto initFunc = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "__init", funcType);

    Block *entryBlock = initFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate statements in init block
    for (const auto &stmt : initNode.children) {
        generateStatement(stmt);
    }

    // Add return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    // Restore insertion point to module level
    builder.setInsertionPointToEnd(module.getBody());
}

void SmolluMLIRGenerator::generateFunctionsBlock(const SmolluASTNode &functionsNode) {
    // Process each function definition
    for (const auto &child : functionsNode.children) {
        if (child.type == "FunctionDefinition") {
            generateFunctionDefinition(child);
        }
    }
}

void SmolluMLIRGenerator::generateFunctionDefinition(const SmolluASTNode &funcDef) {
    std::string funcName = funcDef.value;

    // Clear local variables for new function scope
    clearLocalVars();

    // Parse parameters - they are Identifier nodes before the Block
    std::vector<Type> paramTypes;
    size_t bodyIndex = 0;

    for (size_t i = 0; i < funcDef.children.size(); ++i) {
        if (funcDef.children[i].type == "Block") {
            bodyIndex = i;
            break;
        } else if (funcDef.children[i].type == "Identifier") {
            // Register parameter as local variable
            getOrCreateLocalVar(funcDef.children[i].value);
            paramTypes.push_back(builder.getI32Type()); // Default to i32
        }
    }

    // Create function with return type (assuming i32 for now)
    auto funcType = builder.getFunctionType(paramTypes, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), funcName, funcType);

    Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate function body
    if (bodyIndex < funcDef.children.size()) {
        generateBlock(funcDef.children[bodyIndex]);
    }

    // Add default return if no explicit return
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    // Restore insertion point to module level
    builder.setInsertionPointToEnd(module.getBody());
}

void SmolluMLIRGenerator::generateReturnStatement(const SmolluASTNode &returnStmt) {
    if (returnStmt.children.empty()) {
        // Return void - pass nullptr as value
        builder.create<ReturnOp>(builder.getUnknownLoc(), Value());
    } else {
        // Return value
        Value retValue = generateExpression(returnStmt.children[0]);
        if (retValue) {
            builder.create<ReturnOp>(builder.getUnknownLoc(), retValue);
        } else {
            builder.create<ReturnOp>(builder.getUnknownLoc(), Value());
        }
    }
}

Value SmolluMLIRGenerator::generateFunctionCall(const SmolluASTNode &funcCall) {
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
        builder.getUnknownLoc(),
        builder.getI32Type(),  // Return type (assuming i32)
        mlir::SymbolRefAttr::get(builder.getContext(), funcName),
        args
    );

    return callOp.getResult();
}