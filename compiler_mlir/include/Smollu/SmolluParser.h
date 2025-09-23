//===- SmolluParser.h - Unified Smollu parser interface --------*- C++ -*-===//
//
// Unified API interface for Smollu parsing with AST-only and MLIR modes
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_PARSER_H
#define SMOLLU_PARSER_H

#include "Smollu/SmolluAST.h"
#include "Smollu/SmolluMLIRGen.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <iostream>
#include <string>

namespace mlir {
namespace smollu {

// Compilation modes
enum class CompilationMode {
    AST_ONLY,    // Generate AST only
    MLIR_MODULE  // Generate MLIR module
};

// Main parser interface that provides unified access to both AST-only and MLIR modes
class SmolluParser {
private:
    MLIRContext *context;
    CompilationMode mode;

public:
    SmolluParser(MLIRContext *ctx, CompilationMode m = CompilationMode::MLIR_MODULE);

    // Parse source file and return AST (always available)
    SmolluASTNode parseToAST(const std::string &source);

    // Parse source file and return MLIR module (only in MLIR_MODULE mode)
    ModuleOp parseToMLIR(const std::string &source);

    // Parse source file and emit AST to output stream
    bool parseAndEmitAST(const std::string &source, std::ostream &out = std::cout);

    // Parse source file and return MLIR module with optional AST output
    ModuleOp parseAndEmitMLIR(const std::string &source, bool printAST = false, std::ostream &out = std::cout);

    // Get current compilation mode
    CompilationMode getMode() const { return mode; }

    // Set compilation mode
    void setMode(CompilationMode m) { mode = m; }
};

} // namespace smollu
} // namespace mlir

// Parse Smollu source to MLIR (legacy interface)
mlir::ModuleOp parseSmolluToMLIR(mlir::MLIRContext *context, const char *source);

// Parse Smollu source to AST and emit to stdout
bool parseSmolluToAST(const char *source);

// Parse Smollu source with mode selection
mlir::ModuleOp parseSmolluWithMode(mlir::MLIRContext *context, const char *source, bool emitAST);

#endif // SMOLLU_PARSER_H