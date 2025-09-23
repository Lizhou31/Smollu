//===- SmolluOps.h - Smollu operations header ------------------*- C++ -*-===//
//
// Header for Smollu operations
//
//===----------------------------------------------------------------------===//

#ifndef SMOLLU_OPS_H
#define SMOLLU_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "SmolluOps.h.inc"

#endif // SMOLLU_OPS_H