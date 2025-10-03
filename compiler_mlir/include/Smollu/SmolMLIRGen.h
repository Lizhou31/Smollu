//===- SmolMLIRGen.h - MLIR generator for Smol dialect ---------*- C++ -*-===//
//
// Generates high-level Smol dialect MLIR from Smollu AST
//
//===----------------------------------------------------------------------===//

#ifndef SMOL_MLIR_GEN_H
#define SMOL_MLIR_GEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "Smollu/SmolluAST.h"
#include <map>
#include <string>
#include <iostream>

namespace mlir {
namespace smol {

class SmolMLIRGenerator {
public:
    SmolMLIRGenerator(MLIRContext *ctx);

    /// Generate MLIR module from AST
    ModuleOp generateMLIR(const mlir::smollu::SmolluASTNode &ast);

private:
    // Context and builder
    MLIRContext *context;
    OpBuilder builder;
    ModuleOp module;

    // Variable tracking (for scope verification)
    std::map<std::string, std::string> variableScopes; // name -> "global" or "local"

    // Top-level block generation
    void generateInitBlock(const mlir::smollu::SmolluASTNode &initNode);
    void generateMainBlock(const mlir::smollu::SmolluASTNode &mainNode);
    void generateFunctionsBlock(const mlir::smollu::SmolluASTNode &functionsNode);
    void generateFunctionDefinition(const mlir::smollu::SmolluASTNode &funcDef);

    // Statement generation
    void generateStatement(const mlir::smollu::SmolluASTNode &stmt);
    void generateAssignment(const mlir::smollu::SmolluASTNode &assign, bool isLocal);
    void generateWhileStatement(const mlir::smollu::SmolluASTNode &whileStmt);
    void generateIfStatement(const mlir::smollu::SmolluASTNode &ifStmt);
    void generateNativeCall(const mlir::smollu::SmolluASTNode &nativeCall);
    void generateReturnStatement(const mlir::smollu::SmolluASTNode &returnStmt);
    void generateBlock(const mlir::smollu::SmolluASTNode &block);

    // Expression generation
    Value generateExpression(const mlir::smollu::SmolluASTNode &expr);
    Value generateFunctionCall(const mlir::smollu::SmolluASTNode &funcCall);

    // Helpers
    void clearLocalVars();
    std::string getVarScope(const std::string &name);

    // Location helper - convert AST node location to MLIR location
    Location getLoc(const mlir::smollu::SmolluASTNode &node) {
        return FileLineColLoc::get(builder.getStringAttr(node.filename), node.line, node.column);
    }
};

} // namespace smol
} // namespace mlir

#endif // SMOL_MLIR_GEN_H
