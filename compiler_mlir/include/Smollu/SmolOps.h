//===- SmolOps.h - Smol high-level operations ------------------*- C++ -*-===//
//
// Smol operation declarations - high-level language operations
//
//===----------------------------------------------------------------------===//

#ifndef SMOL_OPS_H
#define SMOL_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "Smollu/SmolDialect.h"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "SmolOps.h.inc"

#endif // SMOL_OPS_H
