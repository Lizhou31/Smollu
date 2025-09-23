//===- Parser.cpp - Smollu parser components integration -------*- C++ -*-===//
//
// Integration file that includes all Smollu parser components
// This file serves as the main entry point for the refactored parser
//
//===----------------------------------------------------------------------===//

// For now, just include the main parser interface
// The other components are built separately
#include "Smollu/SmolluParser.h"

// Legacy C API implementation for backward compatibility
mlir::ModuleOp parseSmolluToMLIR(mlir::MLIRContext *context, const char *source) {
    mlir::smollu::SmolluParser parser(context, mlir::smollu::CompilationMode::MLIR_MODULE);
    return parser.parseAndEmitMLIR(std::string(source), true); // Print AST for legacy behavior
}