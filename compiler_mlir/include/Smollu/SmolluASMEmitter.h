//===- SmolluASMEmitter.h - MLIR to Smollu ASM emitter ---------*- C++ -*-===//
//
// Emits human-readable Smollu assembly from MLIR representation
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_ASM_EMITTER_H
#define SMOLLU_ASM_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir {
namespace smollu {

/// Emit Smollu assembly from MLIR module
bool emitASMFromMLIR(mlir::ModuleOp module, const char *outputFile);

} // namespace smollu
} // namespace mlir

#endif // SMOLLU_ASM_EMITTER_H
