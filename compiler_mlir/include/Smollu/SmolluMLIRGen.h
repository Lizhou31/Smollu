//===- SmolluMLIRGen.h - MLIR generator from AST ---------------*- C++ -*-===//
//
// MLIR generator that converts Smollu AST to MLIR representation
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_MLIR_GEN_H
#define SMOLLU_MLIR_GEN_H

#include "Smollu/SmolluAST.h"
#include "Smollu/SmolluDialect.h"
#include "Smollu/SmolluOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <map>

namespace mlir {
namespace smollu {

// MLIR generator that converts AST to MLIR module
class SmolluMLIRGenerator {
private:
    MLIRContext *context;
    OpBuilder builder;
    ModuleOp module;
    std::map<std::string, uint8_t> globalVars;
    std::map<std::string, uint8_t> localVars;
    uint8_t nextGlobalSlot = 0;
    uint8_t nextLocalSlot = 0;

public:
    SmolluMLIRGenerator(MLIRContext *ctx);

    // Generate MLIR module from AST
    ModuleOp generateMLIR(const SmolluASTNode &ast);

private:
    // AST to MLIR conversion methods
    void generateMainBlock(const SmolluASTNode &mainNode);
    void generateStatement(const SmolluASTNode &stmt);
    void generateAssignment(const SmolluASTNode &assign, bool isLocal);
    Value generateExpression(const SmolluASTNode &expr);
    void generateWhileStatement(const SmolluASTNode &whileStmt);
    void generateIfStatement(const SmolluASTNode &ifStmt);
    void generateBlock(const SmolluASTNode &block);
    void generateNativeCall(const SmolluASTNode &nativeCall);

    // Helper methods
    uint8_t getOrCreateGlobalVar(const std::string &name);
    uint8_t getOrCreateLocalVar(const std::string &name);
};

} // namespace smollu
} // namespace mlir

#endif // SMOLLU_MLIR_GEN_H